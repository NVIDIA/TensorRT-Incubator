# RUN: %PYTHON %s 2>&1 | FileCheck %s
# REQUIRES: host-has-at-least-1-gpus
# REQUIRES: cuda-toolkit-major-version-lte-12
import array
import gc

import mlir_tensorrt.runtime.api as runtime
import numpy as np
import cupy as cp

client = runtime.RuntimeClient()
stream = client.create_stream()
devices = client.get_devices()


def create_memref_from_array_gpu(arr, explicit_dtype=None):
    print(f"Array: {arr} dtype={explicit_dtype}")
    memref = client.create_memref(arr, device=devices[0], dtype=explicit_dtype)
    print(
        f"-- MemRefValue shape={memref.shape} dtype={memref.dtype} strides={memref.strides}"
    )
    from_dlpack = cp.from_dlpack(memref)
    print(f"-- Cupy Array: {from_dlpack} dtype={from_dlpack.dtype}")


print(f"Test client.create_memref -> cupy.from_dlpack")
create_memref_from_array_gpu(array.array("f", [5.0, 4.0, 2.0]))
create_memref_from_array_gpu(array.array("i", [7, 8, 9]))
create_memref_from_array_gpu(array.array("b", [1, 2, 3]))
create_memref_from_array_gpu(
    array.array("b", [True, False, True]), explicit_dtype=runtime.ScalarTypeCode.i1
)


# CHECK-LABEL: Test client.create_memref -> cupy.from_dlpack
# CHECK-NEXT: Array: array('f', [5.0, 4.0, 2.0]) dtype=None
# CHECK-NEXT: -- MemRefValue shape=[3] dtype=ScalarTypeCode.f32 strides=[1]
# CHECK-NEXT: -- Cupy Array: [5. 4. 2.] dtype=float32
# CHECK-NEXT: Array: array('i', [7, 8, 9]) dtype=None
# CHECK-NEXT: -- MemRefValue shape=[3] dtype=ScalarTypeCode.i32 strides=[1]
# CHECK-NEXT: -- Cupy Array: [7 8 9] dtype=int32
# CHECK-NEXT: Array: array('b', [1, 2, 3]) dtype=None
# CHECK-NEXT: -- MemRefValue shape=[3] dtype=ScalarTypeCode.i8 strides=[1]
# CHECK-NEXT: -- Cupy Array: [1 2 3] dtype=int8
# CHECK-NEXT: Array: array('b', [1, 0, 1]) dtype=ScalarTypeCode.i1
# CHECK-NEXT: -- MemRefValue shape=[3] dtype=ScalarTypeCode.i1 strides=[1]
# CHECK-NEXT: -- Cupy Array: [ True False  True] dtype=bool


CUPY_TO_MLIR_TRT = {
    cp.bool_: runtime.ScalarTypeCode.i1,
    cp.int8: runtime.ScalarTypeCode.i8,
    cp.int16: runtime.ScalarTypeCode.i16,
    cp.int32: runtime.ScalarTypeCode.i32,
    cp.int64: runtime.ScalarTypeCode.i64,
    cp.uint8: runtime.ScalarTypeCode.ui8,
    cp.float16: runtime.ScalarTypeCode.f16,
    cp.float32: runtime.ScalarTypeCode.f32,
    cp.float64: runtime.ScalarTypeCode.f64,
}


print(f"Test CuPy Array -> client.device_memref_view -> cupy.from_dlpack")


def test_memref_view_from_cupy():
    data = [1, 2, 3]
    for dtype in list(CUPY_TO_MLIR_TRT.keys()):
        arr = cp.array(data, dtype=dtype)
        print(f"Array: {arr} dtype={arr.dtype}")
        print(
            f"Cupy Array ({arr.dtype}): ",
            cp.from_dlpack(
                client.create_device_memref_view(
                    int(arr.data.ptr),
                    shape=list(arr.shape),
                    dtype=CUPY_TO_MLIR_TRT.get(arr.dtype.type),
                    device=devices[0],
                )
            ),
        )


test_memref_view_from_cupy()

# CHECK-LABEL: Test CuPy Array -> client.device_memref_view -> cupy.from_dlpack
#  CHECK-NEXT: Array: [ True  True  True] dtype=bool
#  CHECK-NEXT: Cupy Array (bool):  [ True  True  True]
#  CHECK-NEXT: Array: [1 2 3] dtype=int8
#  CHECK-NEXT: Cupy Array (int8):  [1 2 3]
#  CHECK-NEXT: Array: [1 2 3] dtype=int16
#  CHECK-NEXT: Cupy Array (int16):  [1 2 3]
#  CHECK-NEXT: Array: [1 2 3] dtype=int32
#  CHECK-NEXT: Cupy Array (int32):  [1 2 3]
#  CHECK-NEXT: Array: [1 2 3] dtype=int64
#  CHECK-NEXT: Cupy Array (int64):  [1 2 3]
#  CHECK-NEXT: Array: [1 2 3] dtype=uint8
#  CHECK-NEXT: Cupy Array (uint8):  [1 2 3]
#  CHECK-NEXT: Array: [1. 2. 3.] dtype=float16
#  CHECK-NEXT: Cupy Array (float16):  [1. 2. 3.]
#  CHECK-NEXT: Array: [1. 2. 3.] dtype=float32
#  CHECK-NEXT: Cupy Array (float32):  [1. 2. 3.]
#  CHECK-NEXT: Array: [1. 2. 3.] dtype=float64
#  CHECK-NEXT: Cupy Array (float64):  [1. 2. 3.]


def check_non_canonical_stride(client, assert_canonical_strides):
    try:
        t = cp.arange(12, dtype=cp.float32).reshape(3, 4)
        a = cp.transpose(t)
        memref = client.create_memref_view_from_dlpack(
            a.__dlpack__(), assert_canonical_strides
        )
    except Exception as e:
        print(f"Received error message: {str(e)}")


def check_canonical_stride(client, assert_canonical_strides):
    try:
        t = cp.arange(12, dtype=cp.float32).reshape(3, 4)
        memref = client.create_memref_view_from_dlpack(
            t.__dlpack__(), assert_canonical_strides
        )
    except Exception as e:
        print(f"Received error message: {str(e)}")


def test_memref_strides():
    print("Testing non-canonical stride: assert_canonical_strides = True")
    non_canonical_result = check_non_canonical_stride(
        client, assert_canonical_strides=True
    )

    print("Testing non-canonical stride: assert_canonical_strides = False")
    non_canonical_result = check_non_canonical_stride(
        client, assert_canonical_strides=False
    )

    print("Testing canonical stride: assert_canonical_strides = True")
    canonical_result = check_canonical_stride(client, assert_canonical_strides=True)

    print("Testing canonical stride: assert_canonical_strides = False")
    canonical_result = check_canonical_stride(client, assert_canonical_strides=False)


print("Test memref strides")
test_memref_strides()

# CHECK-LABEL: Test memref strides
# CHECK-NEXT: Testing non-canonical stride: assert_canonical_strides = True
# CHECK-NEXT: Received error message: InvalidArgument: InvalidArgument:
# CHECK-SAME: Given strides [1, 4] do not match canonical strides [3, 1] for shape [4, 3]
# CHECK-NEXT: Testing non-canonical stride: assert_canonical_strides = False
# CHECK-NEXT: Testing canonical stride: assert_canonical_strides = True
# CHECK-NEXT: Testing canonical stride: assert_canonical_strides = False


def test_memref_allocations():
    # External gpu allocation and gpu view
    data = cp.ones((1000,), dtype=cp.int32)
    memref = client.create_device_memref_view(
        data.data.ptr,
        shape=list(data.shape),
        dtype=CUPY_TO_MLIR_TRT[data.dtype.type],
        device=devices[0],
    )
    assert data.data.ptr == memref.ptr

    # External gpu allocation and cpu view, does not throw an exception and defer usage to memref user.
    data = cp.ones((1000,), dtype=cp.int32)
    memref = client.create_host_memref_view(
        data.data.ptr, shape=list(data.shape), dtype=CUPY_TO_MLIR_TRT[data.dtype.type]
    )
    assert data.data.ptr == memref.ptr


test_memref_allocations()
