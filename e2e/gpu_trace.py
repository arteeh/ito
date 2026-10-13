"""Host-side map phases and CuPy transfers, paired with gpu_cupti hardware activities."""

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter


class MapTrace:
    def __init__(self):
        self.transfers = []
        self.phases = defaultdict(list)

    @contextmanager
    def record(self, backend):
        import cupy
        import numpy as np

        originals = {name: getattr(backend, name) for name in ("integrate_points", "carve")}
        copy_out, copy_in = cupy.asnumpy, cupy.asarray

        def output(array, *args, **kwargs):
            began = perf_counter()
            result = copy_out(array, *args, **kwargs)
            self.transfers.append(
                dict(
                    direction="D2H", bytes=int(array.nbytes), cpu_ms=(perf_counter() - began) * 1000
                )
            )
            return result

        def input(array, *args, **kwargs):
            began = perf_counter()
            result = copy_in(array, *args, **kwargs)
            if isinstance(array, np.ndarray):
                self.transfers.append(
                    dict(
                        direction="H2D",
                        bytes=int(array.nbytes),
                        cpu_ms=(perf_counter() - began) * 1000,
                    )
                )
            return result

        def phase(name):
            def run(*args, **kwargs):
                began = perf_counter()
                result = originals[name](*args, **kwargs)
                self.phases[name].append((perf_counter() - began) * 1000)
                return result

            return run

        try:
            cupy.asnumpy, cupy.asarray = output, input
            for name in originals:
                setattr(backend, name, phase(name))
            yield
        finally:
            cupy.asnumpy, cupy.asarray = copy_out, copy_in
            for name, original in originals.items():
                setattr(backend, name, original)

    def report(self):
        return dict(
            scope="host map phases and transfer calls; default blocking D2H copies",
            map_phases_ms=dict(self.phases),
            cupy_transfers=self.transfers,
        )
