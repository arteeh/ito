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


class Cupti:
    def __init__(self):
        import torch

        path = Path(torch.__file__).parent / "lib/cupti64_2024.1.0.dll"
        self.dll = c.CDLL(str(path))
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
        self.check(self.dll.cuptiActivityEnable(10))
        self.check(self.dll.cuptiActivityEnable(1))
        return self

    def __exit__(self, *_):
        import torch

        torch.cuda.synchronize()
        self.check(self.dll.cuptiActivityFlushAll(1))
        self.check(self.dll.cuptiActivityDisable(10))
        self.check(self.dll.cuptiActivityDisable(1))
        if self.errors or self.dropped:
            raise RuntimeError(
                f"Incomplete CUPTI trace: errors={self.errors}, dropped={self.dropped}"
            )

    def report(self):
        return dict(
            scope="CUPTI concurrent kernel and memcpy activities; hardware timestamps; "
            "diagnostic run, not acceptance timings",
            api_version=self.version,
            dropped=self.dropped,
            activities=self.rows,
        )
