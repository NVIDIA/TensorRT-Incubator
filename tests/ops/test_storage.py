from tripy.ops import Storage

import tripy
import numpy as np
import cupy as cp


class TestStorage:
    def test_cpu_storage(self):
        storage = Storage([1, 2, 3], tripy.device("cpu"))
        assert isinstance(storage.data, np.ndarray)
        assert storage.device.kind == "cpu"

    def test_gpu_storage(self):
        storage = Storage([1, 2, 3], tripy.device("gpu"))
        assert isinstance(storage.data, cp.ndarray)
        assert storage.device.kind == "gpu"
