"""CUDA 12.4 CUPTI activity records on Windows; no kernel serialization.

The official Python bindings cover Linux. These two record prefixes follow
CUDA 12.4's cupti_activity.h; the DLL comes from the pinned Torch bundle.
"""

import ctypes as c
from pathlib import Path


class Kernel(c.Structure):
    _fields_ = [
        ("kind", c.c_uint32),
        ("cache", c.c_uint8),
        ("shared", c.c_uint8),
        ("registers", c.c_uint16),
        ("cache_requested", c.c_uint32),
        ("cache_executed", c.c_uint32),
        ("start", c.c_uint64),
        ("end", c.c_uint64),
        ("completed", c.c_uint64),
        *[(name, c.c_uint32) for name in ("device", "context", "stream")],
        *[
            (name, c.c_int32)
            for name in (
                "grid_x",
                "grid_y",
                "grid_z",
                "block_x",
                "block_y",
                "block_z",
                "static_shared",
                "dynamic_shared",
            )
        ],
        *[(name, c.c_uint32) for name in ("local_per_thread", "local_total", "correlation")],
        ("grid_id", c.c_int64),
        ("name", c.c_char_p),
    ]


class Copy(c.Structure):
    _fields_ = [
        ("kind", c.c_uint32),
        ("direction", c.c_uint8),
        ("source", c.c_uint8),
        ("destination", c.c_uint8),
        ("flags", c.c_uint8),
        ("bytes", c.c_uint64),
        ("start", c.c_uint64),
        ("end", c.c_uint64),
        *[(name, c.c_uint32) for name in ("device", "context", "stream", "correlation")],
    ]


class API(c.Structure):
    _fields_ = [
        ("kind", c.c_uint32),
        ("callback", c.c_uint32),
        ("start", c.c_uint64),
        ("end", c.c_uint64),
        *[(name, c.c_uint32) for name in ("process", "thread", "correlation", "result")],
    ]


class Cupti:
    def __init__(self, *, apis=False, allocations_only=False):
        import torch

        path = Path(torch.__file__).parent / "lib/cupti64_2024.1.0.dll"
        self.dll = c.CDLL(str(path))
        self.apis = apis
        self.allocations_only = allocations_only
        self.api_names = {}
        self.kinds = (4, 5) if allocations_only else (10, 1) + ((4, 5) if apis else ())
        self.buffers = {}
        self.rows = []
        self.errors = []
        self.dropped = 0
        version = c.c_uint32()
        self.check(self.dll.cuptiGetVersion(c.byref(version)))
        self.version = version.value
        if self.version != 22:
            raise RuntimeError(f"Expected CUPTI API 22 (CUDA 12.4), found {self.version}")
        self.dll.cuptiActivityGetNextRecord.argtypes = [
            c.c_void_p,
            c.c_size_t,
            c.POINTER(c.c_void_p),
        ]
        self.dll.cuptiActivityGetNumDroppedRecords.argtypes = [
            c.c_void_p,
            c.c_uint32,
            c.POINTER(c.c_size_t),
        ]
        request = c.CFUNCTYPE(
            None, c.POINTER(c.c_void_p), c.POINTER(c.c_size_t), c.POINTER(c.c_size_t)
        )
        complete = c.CFUNCTYPE(None, c.c_void_p, c.c_uint32, c.c_void_p, c.c_size_t, c.c_size_t)
        self.request = request(self.requested)
        self.complete = complete(self.completed)
        self.check(self.dll.cuptiActivityRegisterCallbacks(self.request, self.complete))

    @staticmethod
    def check(result):
        if result:
            raise RuntimeError(f"CUPTI returned {result}")

    def requested(self, pointer, size, maximum):
        try:
            buffer = c.create_string_buffer(4 * 1024**2 + 8)
            address = (c.addressof(buffer) + 7) & ~7
            self.buffers[address] = buffer
            pointer[0], size[0], maximum[0] = address, 4 * 1024**2, 0
        except Exception as exc:
            self.errors.append(str(exc))
            pointer[0] = None

    def completed(self, context, stream, buffer, size, valid):
        try:
            record = c.c_void_p()
            while True:
                result = self.dll.cuptiActivityGetNextRecord(buffer, valid, c.byref(record))
                if result == 12:  # CUPTI_ERROR_MAX_LIMIT_REACHED: buffer exhausted.
                    break
                self.check(result)
                kind = c.c_uint32.from_address(record.value).value
                if kind == 10:
                    value = Kernel.from_address(record.value)
                    self.rows.append(
                        dict(
                            kind="kernel",
                            name=value.name.decode("utf8"),
                            start_ns=value.start,
                            end_ns=value.end,
                            correlation=value.correlation,
                            stream=value.stream,
                        )
                    )
                elif kind in (4, 5):
                    value = API.from_address(record.value)
                    key = (kind, value.callback)
                    if key not in self.api_names:
                        name = c.c_char_p()
                        self.check(
                            self.dll.cuptiGetCallbackName(
                                1 if kind == 4 else 2, value.callback, c.byref(name)
                            )
                        )
                        self.api_names[key] = name.value.decode()
                    name = self.api_names[key]
                    if not self.allocations_only or any(
                        word in name.lower() for word in ("alloc", "free")
                    ):
                        self.rows.append(
                            dict(
                                kind="api",
                                name=name,
                                start_ns=value.start,
                                end_ns=value.end,
                                correlation=value.correlation,
                            )
                        )
                elif kind == 1:
                    value = Copy.from_address(record.value)
                    self.rows.append(
                        dict(
                            kind="copy",
                            direction=value.direction,
                            bytes=value.bytes,
                            start_ns=value.start,
                            end_ns=value.end,
                            correlation=value.correlation,
                            stream=value.stream,
                        )
                    )
            dropped = c.c_size_t()
            self.check(
                self.dll.cuptiActivityGetNumDroppedRecords(context, stream, c.byref(dropped))
            )
            self.dropped += dropped.value
        except Exception as exc:
            self.errors.append(str(exc))
        finally:
            self.buffers.pop(buffer, None)

    def __enter__(self):
        import time

        timestamp = c.c_uint64()
        self.check(self.dll.cuptiGetTimestamp(c.byref(timestamp)))
        self.anchor = dict(cupti_ns=timestamp.value, perf_ns=time.perf_counter_ns())
        for kind in self.kinds:
            self.check(self.dll.cuptiActivityEnable(kind))
        return self

    def __exit__(self, *_):
        import torch

        torch.cuda.synchronize()
        self.check(self.dll.cuptiActivityFlushAll(1))
        for kind in self.kinds:
            self.check(self.dll.cuptiActivityDisable(kind))
        if self.errors or self.dropped:
            raise RuntimeError(
                f"Incomplete CUPTI trace: errors={self.errors}, dropped={self.dropped}"
            )

    def report(self):
        return dict(
            scope="CUPTI concurrent kernel and memcpy activities; hardware timestamps; "
            "diagnostic run, not acceptance timings",
            api_version=self.version,
            clock_anchor=self.anchor,
            dropped=self.dropped,
            activities=self.rows,
        )
