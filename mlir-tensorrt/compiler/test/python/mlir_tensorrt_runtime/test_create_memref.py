# RUN: %PYTHON %s 2>&1 | FileCheck %s
# REQUIRES: host-has-at-least-1-gpus

import array
import gc

import mlir_tensorrt.runtime.api as runtime
import numpy as np

client = runtime.RuntimeClient()
stream = client.get_devices()[0].stream
devices = client.get_devices()


def create_memref_cpu(shape, dtype):
    memref = client.create_memref(shape=shape, dtype=dtype)
    print(f"CPU Memref shape: {memref.shape}")
    print(f"CPU Memref dtype: {memref.dtype}")


print(f"Test MemRefValue CPU allocation")
create_memref_cpu((1, 2, 3), runtime.ScalarTypeCode.f16)
create_memref_cpu((1, 2, 3), runtime.ScalarTypeCode.i8)
create_memref_cpu((1, 2, 3), runtime.ScalarTypeCode.i1)

# CHECK-LABEL: Test MemRefValue CPU allocation
# CHECK: CPU Memref shape: [1, 2, 3]
# CHECK: CPU Memref dtype: ScalarTypeCode.f16
# CHECK: CPU Memref shape: [1, 2, 3]
# CHECK: CPU Memref dtype: ScalarTypeCode.i8
# CHECK: CPU Memref shape: [1, 2, 3]
# CHECK: CPU Memref dtype: ScalarTypeCode.i1


def create_memref_gpu(shape, dtype):
    memref = client.create_memref(shape=shape, dtype=dtype, device=devices[0])
    print(f"GPU Memref shape: {memref.shape}")
    print(f"GPU Memref dtype: {memref.dtype}")


print(f"Test MemRefValue GPU allocation")
create_memref_gpu((1, 2, 3), runtime.ScalarTypeCode.bf16)
create_memref_gpu((1, 2, 3), runtime.ScalarTypeCode.f32)
create_memref_gpu((1, 2, 3), runtime.ScalarTypeCode.i1)

# CHECK-LABEL: Test MemRefValue GPU allocation
# CHECK: GPU Memref shape: [1, 2, 3]
# CHECK: GPU Memref dtype: ScalarTypeCode.bf16
# CHECK: GPU Memref shape: [1, 2, 3]
# CHECK: GPU Memref dtype: ScalarTypeCode.f32
# CHECK: GPU Memref shape: [1, 2, 3]
# CHECK: GPU Memref dtype: ScalarTypeCode.i1


def create_memref_from_array_cpu(arr, explicit_dtype=None):
    print(f"Array: {arr} explicit_type={explicit_dtype}")
    memref = client.create_memref(arr, dtype=explicit_dtype)
    print(
        f"-- MemRefValue shape={memref.shape} dtype={memref.dtype} strides={memref.strides}"
    )
    from_dlpack = np.from_dlpack(memref)
    print(f"-- np.from_dlpack: {from_dlpack} dtype={from_dlpack.dtype}")
    asarray = np.asarray(memref)
    print(f"-- np.asarray: {asarray} dtype={asarray.dtype}")


print(f"Test client.create_memref -> np.(from_dlpack|asarray)")
create_memref_from_array_cpu(array.array("f", [5.0, 4.0, 2.0]))
create_memref_from_array_cpu(array.array("i", [7, 8, 9]))
create_memref_from_array_cpu(array.array("b", [1, 2, 3]))
create_memref_from_array_cpu(array.array("b", [True, False, True]))
create_memref_from_array_cpu(
    array.array("b", [True, False, True]), explicit_dtype=runtime.ScalarTypeCode.i1
)


# CHECK-LABEL: Test client.create_memref -> np.(from_dlpack|asarray)
#  CHECK-NEXT: Array: array('f', [5.0, 4.0, 2.0]) explicit_type=None
#  CHECK-NEXT: -- MemRefValue shape=[3] dtype=ScalarTypeCode.f32 strides=[1]
#  CHECK-NEXT: -- np.from_dlpack: [5. 4. 2.] dtype=float32
#  CHECK-NEXT: -- np.asarray: [5. 4. 2.] dtype=float32
#  CHECK-NEXT: Array: array('i', [7, 8, 9]) explicit_type=None
#  CHECK-NEXT: -- MemRefValue shape=[3] dtype=ScalarTypeCode.i32 strides=[1]
#  CHECK-NEXT: -- np.from_dlpack: [7 8 9] dtype=int32
#  CHECK-NEXT: -- np.asarray: [7 8 9] dtype=int32
#  CHECK-NEXT: Array: array('b', [1, 2, 3]) explicit_type=None
#  CHECK-NEXT: -- MemRefValue shape=[3] dtype=ScalarTypeCode.i8 strides=[1]
#  CHECK-NEXT: -- np.from_dlpack: [1 2 3] dtype=int8
#  CHECK-NEXT: -- np.asarray: [1 2 3] dtype=int8
#  CHECK-NEXT: Array: array('b', [1, 0, 1]) explicit_type=None
#  CHECK-NEXT: -- MemRefValue shape=[3] dtype=ScalarTypeCode.i8 strides=[1]
#  CHECK-NEXT: -- np.from_dlpack: [1 0 1] dtype=int8
#  CHECK-NEXT: -- np.asarray: [1 0 1] dtype=int8
#  CHECK-NEXT: Array: array('b', [1, 0, 1]) explicit_type=ScalarTypeCode.i1
#  CHECK-NEXT: -- MemRefValue shape=[3] dtype=ScalarTypeCode.i1 strides=[1]
#  CHECK-NEXT: -- np.from_dlpack: [ True False  True] dtype=bool
#  CHECK-NEXT: -- np.asarray: [ True False  True] dtype=bool

NUMPY_TO_MLIR_TRT = {
    np.bool_: runtime.ScalarTypeCode.i1,
    np.int8: runtime.ScalarTypeCode.i8,
    np.int16: runtime.ScalarTypeCode.i16,
    np.int32: runtime.ScalarTypeCode.i32,
    np.int64: runtime.ScalarTypeCode.i64,
    np.uint8: runtime.ScalarTypeCode.ui8,
    np.float16: runtime.ScalarTypeCode.f16,
    np.float32: runtime.ScalarTypeCode.f32,
    np.float64: runtime.ScalarTypeCode.f64,
    np.complex64: runtime.ScalarTypeCode.complex32,
    np.complex128: runtime.ScalarTypeCode.complex64,
}


print(f"Test NumPy Array -> client.create_host_memref_view -> np.from_dlpack")


def test_memref_view_from_numpy():
    data = [1, 2, 3]
    for dtype in list(NUMPY_TO_MLIR_TRT.keys()):
        arr = np.array(data, dtype=dtype)
        print(f"Array: {arr} dtype={arr.dtype}")
        print(
            f"Numpy Array ({arr.dtype}): ",
            np.from_dlpack(
                client.create_host_memref_view(
                    int(arr.ctypes.data),
                    shape=list(arr.shape),
                    dtype=NUMPY_TO_MLIR_TRT[arr.dtype.type],
                )
            ),
        )


test_memref_view_from_numpy()

# CHECK-LABEL: Test NumPy Array -> client.create_host_memref_view -> np.from_dlpack
#  CHECK-NEXT: Array: [ True  True  True] dtype=bool
#  CHECK-NEXT: Numpy Array (bool):  [ True  True  True]
#  CHECK-NEXT: Array: [1 2 3] dtype=int8
#  CHECK-NEXT: Numpy Array (int8):  [1 2 3]
#  CHECK-NEXT: Array: [1 2 3] dtype=int16
#  CHECK-NEXT: Numpy Array (int16):  [1 2 3]
#  CHECK-NEXT: Array: [1 2 3] dtype=int32
#  CHECK-NEXT: Numpy Array (int32):  [1 2 3]
#  CHECK-NEXT: Array: [1 2 3] dtype=int64
#  CHECK-NEXT: Numpy Array (int64):  [1 2 3]
#  CHECK-NEXT: Array: [1 2 3] dtype=uint8
#  CHECK-NEXT: Numpy Array (uint8):  [1 2 3]
#  CHECK-NEXT: Array: [1. 2. 3.] dtype=float16
#  CHECK-NEXT: Numpy Array (float16):  [1. 2. 3.]
#  CHECK-NEXT: Array: [1. 2. 3.] dtype=float32
#  CHECK-NEXT: Numpy Array (float32):  [1. 2. 3.]
#  CHECK-NEXT: Array: [1. 2. 3.] dtype=float64
#  CHECK-NEXT: Numpy Array (float64):  [1. 2. 3.]
#  CHECK-NEXT: Array: [1.+0.j 2.+0.j 3.+0.j] dtype=complex64
#  CHECK-NEXT: Numpy Array (complex64):  [1.+0.j 2.+0.j 3.+0.j]
#  CHECK-NEXT: Array: [1.+0.j 2.+0.j 3.+0.j] dtype=complex128
#  CHECK-NEXT: Numpy Array (complex128):  [1.+0.j 2.+0.j 3.+0.j]


def test_memref_allocations():
    # External cpu allocation and cpu view
    data = np.ones((1000,), dtype=np.int32)
    memref = client.create_host_memref_view(
        data.ctypes.data,
        shape=list(data.shape),
        dtype=NUMPY_TO_MLIR_TRT[data.dtype.type],
    )
    assert data.ctypes.data == memref.ptr

    # External cpu allocation and gpu view, does not throw an exception and defer usage to memref user.
    data = np.ones((1000,), dtype=np.int32)
    memref = client.create_device_memref_view(
        data.ctypes.data,
        shape=list(data.shape),
        dtype=NUMPY_TO_MLIR_TRT[data.dtype.type],
        device=devices[0],
    )
    assert data.ctypes.data == memref.ptr

    # Internal allocation and host to host copy
    data = array.array("i", [1] * 1000)
    memref = client.create_memref(data, dtype=runtime.ScalarTypeCode.i32)
    assert data.buffer_info()[0] != memref.ptr

    # Internal allocation and host to device copy
    data = array.array("i", [1] * 1000)
    memref = client.create_memref(
        data, dtype=runtime.ScalarTypeCode.i32, device=devices[0]
    )
    assert data.buffer_info()[0] != memref.ptr


test_memref_allocations()
