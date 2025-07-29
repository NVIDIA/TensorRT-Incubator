# RUN: %PYTHON %s
# REQUIRES: host-has-at-least-1-gpus
# REQUIRES: cuda-toolkit-major-version-lte-12

# Tests Cupy <- DLPack -> MemRefValue conversions.

import array
import gc
import os
import sys
import cupy as cp
import numpy as np
import mlir_tensorrt.runtime.api as runtime

client = runtime.RuntimeClient()
stream = client.create_stream()
devices = client.get_devices()


def roundtrip_framework_buffer_dlpack(framework_array1, module, expected_type_code):
    """Create an array in the framework, convert to MemRefValue via DLPack, and then go back."""
    memref = client.create_memref_view_from_dlpack(framework_array1.__dlpack__())
    assert memref.dtype == expected_type_code, "incorrect dtype"
    assert list(memref.shape) == list(framework_array1.shape), "incorrect shape"
    framework_array2 = module.from_dlpack(memref)
    assert memref.ref_count() == 2, "incorrect ref count"
    if isinstance(framework_array1, np.ndarray):
        np.testing.assert_array_equal(framework_array1, framework_array2)
    elif isinstance(framework_array1, cp.ndarray):
        cp.testing.assert_array_equal(framework_array1, framework_array2)


# CuPy's DLPack support is experimental, so we only run these tests if CuPy is available.
roundtrip_framework_buffer_dlpack(
    np.array([1, 2, 3, 4], dtype=np.int32), np, runtime.ScalarTypeCode.i32
)
roundtrip_framework_buffer_dlpack(
    np.ones((1, 2, 3), dtype=np.float32), np, runtime.ScalarTypeCode.f32
)
roundtrip_framework_buffer_dlpack(
    np.ones(0, dtype=np.int8), np, runtime.ScalarTypeCode.i8
)
roundtrip_framework_buffer_dlpack(
    cp.array([1, 2, 3, 4], dtype=cp.int32), cp, runtime.ScalarTypeCode.i32
)
roundtrip_framework_buffer_dlpack(
    cp.ones((1, 2, 3), dtype=cp.float32), cp, runtime.ScalarTypeCode.f32
)
roundtrip_framework_buffer_dlpack(
    cp.ones(0, dtype=cp.int8), cp, runtime.ScalarTypeCode.i8
)
