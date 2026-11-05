"""Performance trace: timed spans of the pipeline's stages, in every process, with diagnostic
logging on (off, a span costs a flag check).

Each process writes trace/<process>-<pid>.jsonl beside diagnostics.jsonl, a line a span:
{"n": name, "t": start, "d": duration (ns, perf_counter: system-wide on Windows, so processes
line up), "p": process, "th": thread, "k": the frame or map it belongs to, fields...}. GPU
spans carry "gpu": true (Vulkan timestamps or CUDA events: GPU time, not the CPU's wait).
tools/trace.py makes the files a per-stage table and a Chrome/Perfetto trace.
"""

import json
import os
import threading
from pathlib import Path
from time import perf_counter_ns

FLUSH_NS = 500_000_000
ROTATE_BYTES = 64 * 2**20

_on = _cleaned = False
_directory = _name = _file = None
_lines = []
_flushed = 0
_lock = threading.Lock()


def config():
    """Where a child process traces as this one does (start(config(), name)), or None."""
    return str(_directory) if _on else None


def start(directory, name=None, *, fresh=False):
    """Trace this process into `directory` (None: stop). `fresh` (the app): the first start
    removes the traces of earlier runs."""
    global _on, _directory, _name, _file, _cleaned
    stop()
    if directory is None:
        return
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    if fresh and not _cleaned:
        _cleaned = True
        for old in folder.glob("*.jsonl*"):
            try:
                old.unlink()
            except OSError:
                pass
    _name = name or "process"
    _directory = folder
    _file = open(folder / f"{_name}-{os.getpid()}.jsonl", "a", encoding="utf-8")
    _on = True


def stop():
    global _on, _file
    _on = False
    with _lock:
        _flush()
        if _file is not None:
            _file.close()
            _file = None


def enabled():
    return _on


def add(name, start, end, key=None, **fields):
    """A span from perf_counter_ns() `start` to `end`."""
    if not _on:
        return
    record = {"n": name, "t": start, "d": end - start, "p": _name, "th": threading.get_ident()}
    if key is not None:
        record["k"] = key
    record.update(fields)
    line = json.dumps(record, separators=(",", ":"))
    with _lock:
        _lines.append(line)
        if end - _flushed > FLUSH_NS or len(_lines) >= 2048:
            _flush()


def _flush():
    global _flushed, _file
    _flushed = perf_counter_ns()
    if _file is None or not _lines:
        _lines.clear()
        return
    _file.write("\n".join(_lines) + "\n")
    _file.flush()
    _lines.clear()
    if _file.tell() > ROTATE_BYTES:
        path = Path(_file.name)
        _file.close()
        path.replace(path.with_suffix(".jsonl.1"))
        _file = open(path, "a", encoding="utf-8")


class _Span:
    __slots__ = ("name", "key", "fields", "start")

    def __init__(self, name, key, fields):
        self.name, self.key, self.fields = name, key, fields

    def __enter__(self):
        self.start = perf_counter_ns()
        return self

    def __exit__(self, *_):
        add(self.name, self.start, perf_counter_ns(), self.key, **self.fields)


class _Off:
    __slots__ = ()
    fields = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


_OFF = _Off()


def span(name, key=None, **fields):
    """with span("stage", key): ... times the block (its .fields may be added to inside)."""
    return _Span(name, key, fields) if _on else _OFF


def key(seconds):
    """A frame's id in spans: its capture time (pilot clock) in microseconds."""
    return None if seconds is None else int(round(seconds * 1e6))
