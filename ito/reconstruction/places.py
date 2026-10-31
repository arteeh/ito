"""Keep places past Map size (proof of concept, #79).

Upstream MASt3R-SLAM's keyframe buffer is fixed (Map size); feat/slam's full map then maps nothing
new. Here a full map moves its least recently used keyframe out of the GPU buffer to a store on
disk, and brings a stored keyframe back when retrieval recognises its place:

- Every keyframe gets a lifelong id (gid). Retrieval keeps every keyframe it was given (its ids map
  to gids), the splat map owns splats by gid, and a stored keyframe's splats stay in the map as they
  were (they no longer move with the solve) until it comes back.
- Out (the frontend, between the backend's tasks, never waiting for it): the keyframe's buffers go
  to disk, its graph edges go, and the slot is filled by moving the keyframe before the last into it
  and the last one down, so the last keyframe (the tracker's) and the one before (the next
  keyframe's consecutive edge) stay where upstream's code looks for them. A keyframe whose removal
  would split the graph is passed over.
- Back: when a new keyframe's retrieval (the backend) recalls stored keyframes, the frontend loads
  them into the slot before the last, and the backend links each to that keyframe (one edge,
  upstream's matching and threshold) and solves: the old place joins the graph again and its
  splats move to its solved pose. A lost camera's retrieval recalls them the same way; the next
  relocalisation attempt can use them.
The store holds at most STORE_MAX keyframes on disk (about 4 MB each); the least recently used go.
"""

import atexit
import os
import shutil
import tempfile
import time
from collections import deque

import numpy as np

STORE_MAX = 512
# Tests only: 0 keeps feat/slam's full map (nothing moves out), the comparison's control.
ENABLED = os.environ.get("ITO_KEEP_PLACES", "1") != "0"
RELOADS = 3  # stored keyframes loaded per frame at most
FIELDS = (
    "dataset_idx",
    "img",
    "uimg",
    "img_shape",
    "img_true_shape",
    "T_WC",
    "X",
    "C",
    "N",
    "N_updates",
    "feat",
    "pos",
    "is_dirty",
)


class Places:
    def __init__(self, backend, buffer):
        self.b = backend
        self.gids = np.full(buffer, -1, np.int64)  # slot -> lifelong keyframe id
        self.used = np.zeros(buffer, np.int64)  # slot -> tick it was last used
        self.tick = 0
        self.next_gid = 0
        self.rid_gid = []  # retrieval id -> gid
        self.directory = tempfile.mkdtemp(prefix="ito-places-")
        atexit.register(shutil.rmtree, self.directory, ignore_errors=True)
        self.stored = {}  # gid -> tick last used (on disk)
        self.wanted = []  # (gid, partner gid or None): stored keyframes retrieval recalled
        self.links = deque()  # (gid, partner gid): reloaded keyframes for the backend to link
        self.starved = set()  # stored gids whose splats the pool evicted meanwhile
        self.evictions = self.reloads = self.links_made = self.links_failed = 0
        self.maintain_seconds = 0.0
        self.maintain_peak = 0.0

    # -- bookkeeping ------------------------------------------------------------------------------
    def touch(self, slots):
        self.tick += 1
        for s in slots:
            self.used[s] = self.tick

    def appended(self):
        """A keyframe was appended at the end (any thread that appends calls this)."""
        s = len(self.b.keyframes) - 1
        self.gids[s] = self.next_gid
        self.next_gid += 1
        self.touch([s])

    def popped(self):
        self.gids[len(self.b.keyframes)] = -1

    def slot(self, gid):
        found = np.flatnonzero(self.gids[: len(self.b.keyframes)] == gid)
        return int(found[0]) if len(found) else None

    def retrieve(self, frame, add_gid, k, min_thresh, partner=None):
        """retrieval.update, in slots: the ids it returns that are in the buffer, and the stored
        ones noted for reloading (with the keyframe that should be linked to them)."""
        retrieval = self.b.retrieval
        rid = retrieval.kf_counter
        found = retrieval.update(
            frame, add_after_query=add_gid is not None, k=k, min_thresh=min_thresh
        )
        if add_gid is not None:
            assert len(self.rid_gid) == rid
            self.rid_gid.append(add_gid)
        slots = []
        for r in found:
            gid = self.rid_gid[r]
            s = self.slot(gid)
            if s is not None:
                slots.append(s)
            elif gid in self.stored and all(gid != w[0] for w in self.wanted):
                self.wanted.append((gid, partner))
        self.touch(slots)
        return slots

    # -- the frontend's turn ----------------------------------------------------------------------
    def maintain(self):
        """Between the backend's tasks: reload what retrieval recalled, keep a slot free."""
        b = self.b
        kf = b.keyframes
        if not ENABLED or (not self.wanted and len(kf) < b.capacity and b.spare is None):
            return
        if not b.backend_lock.acquire(blocking=False):
            return
        began = time.perf_counter()
        try:
            with b.states.lock:
                if b.states.global_optimizer_tasks or self.links:
                    return  # their slots must not move under them
            with kf.lock:
                n = len(kf)
                unshown = set(self.gids[b.shown : n].tolist())
                if b.spare is not None:
                    self.adopt_spare()
                wanted, self.wanted = self.wanted[:RELOADS], []
                for gid, partner in wanted:
                    if self.slot(gid) is not None or gid not in self.stored:
                        continue
                    while len(kf) >= b.capacity:
                        self.evict(self.victim())
                    self.insert(gid)
                    if partner is not None and self.slot(partner) is not None:
                        self.links.append((gid, partner))
                while len(kf) >= b.capacity:
                    self.evict(self.victim())
                self.anchor()
                n = len(kf)
                b.shown = n
                for s in range(n):
                    if self.gids[s] in unshown:
                        kf.is_dirty[s] = True
            b.hand_off()
        finally:
            b.backend_lock.release()
            spent = time.perf_counter() - began
            self.maintain_seconds += spent
            self.maintain_peak = max(self.maintain_peak, spent)

    def adopt_spare(self):
        """feat/slam's spare keyframe (a full map's relocalisation, kept out of retrieval) becomes
        a keyframe like any other."""
        b = self.b
        last = len(b.keyframes) - 1
        rid = b.retrieval.kf_counter
        b.retrieval.update(
            b.keyframes[last],
            add_after_query=True,
            k=b.config["retrieval"]["k"],
            min_thresh=b.config["retrieval"]["min_thresh"],
        )
        assert len(self.rid_gid) == rid
        self.rid_gid.append(int(self.gids[last]))
        b.spare = None

    def victim(self):
        """The least recently used keyframe whose removal does not split the graph; never the
        last two (the tracker's, and the next keyframe's consecutive neighbour)."""
        n = len(self.b.keyframes)
        g = self.b.graph
        ii = g.ii2[: 2 * g.n_edges : 2].tolist()
        jj = g.jj2[: 2 * g.n_edges : 2].tolist()
        order = [int(s) for s in np.argsort(self.used[: n - 2], kind="stable")]
        before = components(ii, jj)
        for s in order:
            kept = [(i, j) for i, j in zip(ii, jj, strict=True) if s != i and s != j]
            if len(kept) == len(ii) or components(*zip(*kept, strict=True)) <= before:
                return s
        return order[0]

    def anchor(self):
        """The oldest keyframe in the buffer into slot 0, which the solve pins (base.yaml pin 1:
        the lowest index in the graph). Upstream pins its first keyframe for good; here the
        first may be in the store, and pinning whichever keyframe happens to sit lowest let the
        buffer's graph drift away from the stored places' poses, so the map doubled when the
        camera came back. Pinning the oldest instead brings the buffer back to an old place
        reloaded into it (it is older than anything the camera mapped since). Slot 0 is never
        the last keyframe, so the tracker's and the next keyframe's neighbour stay."""
        n = len(self.b.keyframes)
        if n < 3:
            return
        s = int(np.argmin(np.where(self.gids[: n - 1] >= 0, self.gids[: n - 1], np.inf)))
        if s == 0:
            return
        spare = n  # free: maintain keeps the buffer below Map size
        self.move(0, spare)
        self.move(s, 0)
        self.move(spare, s)
        self.gids[spare] = -1
        self.renumber({0: s, s: 0})

    def move(self, src, dst):
        b, kf = self.b, self.b.keyframes
        for name in FIELDS:
            tensor = getattr(kf, name)
            tensor[dst] = tensor[src]
        b.drawn[dst] = b.drawn[src]
        b.depth[dst] = b.depth[src]
        for array in (b.pending, b.stale, self.gids, self.used):
            array[dst] = array[src]

    def renumber(self, mapping):
        """Graph edges' keyframe indices through mapping (old slot -> new slot)."""
        torch, g = self.b.torch, self.b.graph
        rows = 2 * g.n_edges
        if not rows or not mapping:
            return
        lut = torch.arange(len(self.gids), device=g.ii2.device)
        for old, new in mapping.items():
            lut[old] = new
        g.ii2[:rows] = lut[g.ii2[:rows]]
        g.jj2[:rows] = lut[g.jj2[:rows]]

    def drop_edges(self, s):
        torch, g = self.b.torch, self.b.graph
        rows = 2 * g.n_edges
        gone = (g.ii2[:rows] == s) | (g.jj2[:rows] == s)
        if not bool(gone.any()):
            return
        keep = torch.nonzero(~gone).flatten()
        # In place, a few rows at a time (idx2 is 1.2 MB a row): keep[i] >= i, so no row is
        # overwritten before it is read.
        for start in range(0, len(keep), 32):
            chosen = keep[start : start + 32]
            for tensor in (g.ii2, g.jj2, g.idx2, g.valid2, g.Q2):
                tensor[start : start + len(chosen)] = tensor[chosen]
        g.n_edges = len(keep) // 2

    def evict(self, s):
        """Keyframe in slot s to the store; the slot gets the keyframe before the last, which gets
        the last."""
        kf = self.b.keyframes
        n = len(kf)
        a, last = n - 2, n - 1
        self.save(s)
        self.drop_edges(s)
        mapping = {}
        if s != a:
            self.move(a, s)
            mapping[a] = s
        self.move(last, a)
        mapping[last] = a
        self.renumber(mapping)
        kf.n_size.value = n - 1
        self.gids[last] = -1
        self.evictions += 1

    def insert(self, gid):
        """Stored keyframe gid into the slot before the last."""
        kf = self.b.keyframes
        n = len(kf)
        last = n - 1
        self.move(last, n)
        self.renumber({last: n})
        kf.n_size.value = n + 1
        self.load(gid, last)
        self.touch([last])
        self.reloads += 1

    # -- the store --------------------------------------------------------------------------------
    def path(self, gid):
        return os.path.join(self.directory, f"{gid}.npz")

    def save(self, s):
        b, kf, torch = self.b, self.b.keyframes, self.b.torch
        gid = int(self.gids[s])
        np.savez(
            self.path(gid),
            dataset_idx=kf.dataset_idx[s].cpu().numpy(),
            uimg=(kf.uimg[s] * 255).round().to(torch.uint8).numpy(),
            img_shape=kf.img_shape[s].cpu().numpy(),
            img_true_shape=kf.img_true_shape[s].cpu().numpy(),
            T_WC=kf.T_WC[s].cpu().numpy(),
            X=kf.X[s].cpu().numpy(),
            C=kf.C[s].cpu().numpy(),
            N=kf.N[s].cpu().numpy(),
            N_updates=kf.N_updates[s].cpu().numpy(),
            feat=kf.feat[s].half().cpu().numpy(),
            pos=kf.pos[s].cpu().numpy(),
            drawn=b.drawn[s].cpu().numpy(),
            depth=b.depth[s].cpu().numpy(),
            pending=np.array(b.pending[s]),
        )
        self.stored[gid] = self.tick
        while len(self.stored) > STORE_MAX:
            oldest = min(self.stored, key=self.stored.get)
            del self.stored[oldest]
            os.remove(self.path(oldest))

    def load(self, gid, s):
        b, kf, torch = self.b, self.b.keyframes, self.b.torch
        with np.load(self.path(gid)) as data:
            fields = {name: data[name] for name in data.files}
        cuda = kf.img.device
        uimg = torch.from_numpy(fields["uimg"]).float() / 255
        kf.uimg[s] = uimg
        # create_frame's ImgNorm of the same 8-bit image, exactly.
        kf.img[s] = (uimg.to(cuda).permute(2, 0, 1) - 0.5) / 0.5
        for name in ("dataset_idx", "img_shape", "img_true_shape", "T_WC", "X", "C", "N"):
            getattr(kf, name)[s] = torch.from_numpy(fields[name]).to(cuda)
        kf.N_updates[s] = torch.from_numpy(fields["N_updates"]).to(cuda)
        kf.feat[s] = torch.from_numpy(fields["feat"]).to(cuda).float()
        kf.pos[s] = torch.from_numpy(fields["pos"]).to(cuda)
        kf.is_dirty[s] = False
        b.drawn[s] = torch.from_numpy(fields["drawn"]).to(cuda)
        b.depth[s] = torch.from_numpy(fields["depth"]).to(cuda)
        b.pending[s] = bool(fields["pending"]) or gid in self.starved
        b.stale[s] = False
        self.starved.discard(gid)
        self.gids[s] = gid
        del self.stored[gid]
        os.remove(self.path(gid))

    # -- the backend's turn -----------------------------------------------------------------------
    def link(self):
        """A reloaded keyframe and the keyframe that recalled it: one edge, and the solve."""
        b = self.b
        gid, partner = self.links[0]
        r, p = self.slot(gid), self.slot(partner)
        if r is None or p is None or 2 * (b.graph.n_edges + 1) > len(b.graph.ii2):
            return
        if b.graph.add_factors([p], [r], b.config["local_opt"]["min_match_frac"]):
            self.links_made += 1
            self.touch([r, p])
            b.graph.solve_GN_calib()
        else:
            self.links_failed += 1

    def status(self):
        return (
            f"{len(self.stored)} stored, {self.evictions} out, {self.reloads} back "
            f"({self.links_made} linked)"
        )


def components(ii=(), jj=()):
    """Connected components of the graph with these edges (over the keyframes they touch): a
    second one has no pinned keyframe, and its gauge is free in the solve."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(ii, jj, strict=True):
        parent[find(i)] = find(j)
    return len({find(x) for x in list(parent)})
