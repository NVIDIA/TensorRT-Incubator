# RUN: %PYTHON %s
# Tests Numpy -> GPU -> Numpy roundtrip with async copies.
import mlir_tensorrt.runtime.api as runtime
import numpy as np

client = runtime.RuntimeClient()
stream = client.create_stream()
devices = client.get_devices()


def check_copy_roundtrip():
    data = np.arange(1000, dtype=np.float32)
    memref = client.create_memref(data, device=devices[0], stream=stream)
    memref = client.copy_to_host(memref, stream=stream)
    # Note: currently we always sync in `copy_to_host`, so removing
    # the below sync won't currently affect the behavior.
    stream.sync()
    np.testing.assert_array_equal(data, np.from_dlpack(memref))


check_copy_roundtrip()
