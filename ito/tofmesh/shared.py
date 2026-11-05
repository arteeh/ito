"""NumPy arrays from one process to another through shared memory, not a pipe.

A mesh of half a million triangles (~20 MB) went through multiprocessing queues: pickled by the
sender's feeder thread, written through a pipe, read and unpickled by the receiver, ~35-45 ms
a hop and CPU on both sides. Here the sender copies the arrays once into a segment of its own
and sends only a small handle; the receiver maps them where they are, and releases the segment
when it is done with them. A segment is the sender's to write again only once released.
"""

import os
import threading
from multiprocessing import shared_memory

import numpy as np

ALIGN = 64


class Arrays:
    """Made by the parent, passed to the sending and the receiving process (or thread)."""

    def __init__(self, context, segments):
        self.busy = context.RawArray("b", segments)  # 1 from send() until release()
        self.memory = [None] * segments  # the sender's segments
        self.opened = {}  # the receiver's: index -> SharedMemory
        self.lock = threading.Lock()

    def __getstate__(self):
        return {"busy": self.busy}

    def __setstate__(self, state):
        self.busy = state["busy"]
        self.memory = [None] * len(self.busy)
        self.opened = {}
        self.lock = threading.Lock()

    def send(self, arrays):
        """Copy the arrays into a free segment: a handle for receive(), or None when every
        segment is still held by the receiver."""
        free = next((i for i, busy in enumerate(self.busy) if not busy), None)
        if free is None:
            return None
        layout, size = [], 0
        for a in arrays:
            layout.append((size, a.dtype.str, a.shape))
            size += -(-a.nbytes // ALIGN) * ALIGN
        memory = self.memory[free]
        if memory is None or memory.size < size:
            if memory is not None:
                _close(memory, owner=True)
            # Room to grow: a mesh grows a little with most updates.
            memory = self.memory[free] = shared_memory.SharedMemory(
                create=True, size=max(ALIGN, int(size * 1.25))
            )
        for (offset, _, _), a in zip(layout, arrays, strict=True):
            view = np.ndarray(a.shape, a.dtype, memory.buf, offset)
            view[...] = a
            del view
        self.busy[free] = 1
        return free, memory.name, layout

    def receive(self, handle):
        """The arrays of a handle, in place (no copy): valid until release(handle)."""
        index, name, layout = handle
        with self.lock:
            memory = self.opened.get(index)
            if memory is None or memory.name != name:
                if memory is not None:
                    _close(memory)
                memory = self.opened[index] = shared_memory.SharedMemory(name=name)
        return [
            np.ndarray(shape, np.dtype(dtype), memory.buf, offset)
            for offset, dtype, shape in layout
        ]

    def release(self, handle):
        self.busy[handle[0]] = 0

    def close(self):
        for memory in self.memory:
            if memory is not None:
                _close(memory, owner=True)
        for memory in self.opened.values():
            _close(memory)


def _close(memory, owner=False):
    try:
        memory.close()
    except BufferError:  # a view still exported: the mapping goes with the process
        return
    if owner and os.name != "nt":
        memory.unlink()


class Lease:
    """A received handle's release once all its holders have let go of its arrays."""

    def __init__(self, arrays, handle):
        self.arrays, self.handle = arrays, handle
        self.holders = 1
        self.lock = threading.Lock()

    def hold(self):
        with self.lock:
            self.holders += 1

    def drop(self):
        with self.lock:
            self.holders -= 1
            done = self.holders == 0
        if done:
            self.arrays.release(self.handle)
