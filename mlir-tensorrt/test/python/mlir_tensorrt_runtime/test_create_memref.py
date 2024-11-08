# RUN: %PYTHON %s 2>&1 | FileCheck %s
# REQUIRES: host-has-at-least-1-gpus

import gc

import array
import cupy as cp
import numpy as np

import mlir_tensorrt.runtime.api as runtime

client = runtime.RuntimeClient()
stream = client.create_stream()
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
}

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

    # External gpu allocation and gpu view
    data = cp.ones((1000,), dtype=cp.int32)
    memref = client.create_device_memref_view(
        data.data.ptr,
        shape=list(data.shape),
        dtype=NUMPY_TO_MLIR_TRT[data.dtype.type],
        device=devices[0],
    )
    assert data.data.ptr == memref.ptr

    # External gpu allocation and cpu view, does not throw an exception and defer usage to memref user.
    data = cp.ones((1000,), dtype=cp.int32)
    memref = client.create_host_memref_view(
        data.data.ptr, shape=list(data.shape), dtype=NUMPY_TO_MLIR_TRT[data.dtype.type]
    )
    assert data.data.ptr == memref.ptr

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


def create_host_memref_external_owner_with_external_reference(arr):
    # memref ptr is owned externally, should not get deleted until np holds memref ptr
    memref = client.create_host_memref_view(
        int(arr.ctypes.data), shape=[3], dtype=runtime.ScalarTypeCode.f64
    )
    return np.from_dlpack(memref)


def create_device_memref_external_owner_with_external_reference(arr):
    # memref ptr is owned externally, should not get deleted until np holds memref ptr
    memref = client.create_device_memref_view(
        int(arr.data.ptr),
        shape=[3],
        dtype=runtime.ScalarTypeCode.f64,
        device=devices[0],
    )
    return cp.from_dlpack(memref)


def create_host_memref_internal_owner_with_external_reference():
    # memref is owned internally, should not get deleted until np holds memref ptr
    arr = array.array("f", [5.0, 4.0, 2.0])
    memref = client.create_memref(arr)
    return np.from_dlpack(memref)


def create_device_memref_internal_owner_with_external_reference():
    # memref is owned internally, should not get deleted until np holds memref ptr
    arr = array.array("f", [5.0, 4.0, 2.0])
    memref = client.create_memref(arr, device=devices[0])
    return cp.from_dlpack(memref)


def test_memref_external_reference():
    np_arr = np.array([5.0, 4.0, 2.0])
    cp_arr = cp.array([5.0, 4.0, 2.0])
    print(
        f"Test externally owned Array -> client.host_memref_view -> externally referenced numpy.from_dlpack"
    )
    print(
        f"Numpy Array: ",
        create_host_memref_external_owner_with_external_reference(np_arr),
    )
    print(
        f"Test externally owned Array -> client.device_memref_view -> externally referenced cupy.from_dlpack"
    )
    print(
        f"Cupy Array: ",
        create_device_memref_external_owner_with_external_reference(cp_arr),
    )
    print(
        f"Test internally owned Array -> client.host_memref -> externally referenced numpy.from_dlpack"
    )
    print(f"Numpy Array: ", create_host_memref_internal_owner_with_external_reference())
    print(
        f"Test internally owned Array -> client.device_memref-> externally referenced cupy.from_dlpack"
    )
    print(
        f"Cupy Array: ", create_device_memref_internal_owner_with_external_reference()
    )


print("Test memref external reference counting")
test_memref_external_reference()

# CHECK-LABEL: Test memref external reference counting
# CHECK-NEXT: Test externally owned Array -> client.host_memref_view -> externally referenced numpy.from_dlpack
# CHECK-NEXT: Numpy Array:  [5. 4. 2.]
# CHECK-NEXT: Test externally owned Array -> client.device_memref_view -> externally referenced cupy.from_dlpack
# CHECK-NEXT: Cupy Array:  [5. 4. 2.]
# CHECK-NEXT: Test internally owned Array -> client.host_memref -> externally referenced numpy.from_dlpack
# CHECK-NEXT: Numpy Array:  [5. 4. 2.]
# CHECK-NEXT: Test internally owned Array -> client.device_memref-> externally referenced cupy.from_dlpack
# CHECK-NEXT: Cupy Array:  [5. 4. 2.]


def test_num_external_references():
    arr = np.array([5.0, 4.0, 2.0])
    memref = client.create_host_memref_view(
        int(arr.ctypes.data), shape=[3], dtype=runtime.ScalarTypeCode.f64
    )
    num_ref_count = 10
    arrays = []
    for i in range(num_ref_count):
        arrays.append(np.from_dlpack(memref))
    print(
        "Number of External reference count: ",
        client.external_reference_count(arr.ctypes.data),
    )


print("Test memref number of external reference counts")
test_num_external_references()

# CHECK-LABEL: Test memref number of external reference counts
# CHECK-NEXT: Number of External reference count:  10


def test_released_internally():
    arr = np.array([5.0, 4.0, 2.0])

    def memref_alloc():
        memref = client.create_host_memref_view(
            int(arr.ctypes.data), shape=[3], dtype=runtime.ScalarTypeCode.f64
        )
        return np.from_dlpack(
            memref
        )  # Ensure we have an externally reference to the pointer.

    _ = memref_alloc()
    print(
        "Memref released internally: ", client.is_released_internally(arr.ctypes.data)
    )


print("Test memref is released internally with an external reference")
test_released_internally()

# CHECK-LABEL: Test memref is released internally with an external reference
# CHECK-NEXT: Memref released internally:  True


def test_memref_lifetime():
    def memref_alloc():
        # allocate internally owned array
        arr_in = array.array("f", [5.0, 4.0, 2.0])
        memref = client.create_memref(arr_in)
        # create an external reference
        arr_out = np.from_dlpack(memref)
        # memref goes out of scope
        return arr_out

    arr_out = memref_alloc()
    gc.collect()  # Ensure memref is GC'ed.
    print(
        "Memref released internally: ",
        client.is_released_internally(arr_out.ctypes.data),
    )
    print(
        "Number of External reference count: ",
        client.external_reference_count(arr_out.ctypes.data),
    )
    print("Numpy Array: ", arr_out)


print("Test memref lifetime")
test_memref_lifetime()

# CHECK-LABEL: Test memref lifetime
# CHECK-NEXT: Memref released internally:  True
# CHECK-NEXT: Number of External reference count:  1
# CHECK-NEXT: Numpy Array:  [5. 4. 2.]


def create_memref_from_dlpack(arr, module):
    print(f"Array: {arr}")
    memref = client.create_memref_view_from_dlpack(arr.__dlpack__())
    print(f"-- Memref shape: {memref.shape}")
    print(f"-- Memref dtype: {memref.dtype}")
    print(f"-- {module.__name__}.from_dlpack(): {module.from_dlpack(memref)}")


print(f"Test np.array -> client.create_memref_from_dlpack")
create_memref_from_dlpack(np.array([1, 2, 3, 4], dtype=np.int32), np)
create_memref_from_dlpack(np.ones((1, 2, 3), dtype=np.float32), np)
create_memref_from_dlpack(np.ones(0, dtype=np.int8), np)
print(f"Test cp.array -> client.create_memref_from_dlpack")
create_memref_from_dlpack(cp.array([1, 2, 3, 4], dtype=cp.int32), cp)
create_memref_from_dlpack(cp.ones((1, 2, 3), dtype=cp.float32), cp)
create_memref_from_dlpack(cp.ones(0, dtype=cp.int8), cp)


# CHECK-LABEL: Test np.array -> client.create_memref_from_dlpack
# CHECK-NEXT: Array: [1 2 3 4]
# CHECK-NEXT: -- Memref shape: [4]
# CHECK-NEXT: -- Memref dtype: ScalarTypeCode.i32
# CHECK-NEXT: -- numpy.from_dlpack(): [1 2 3 4]
# CHECK-NEXT: Array: {{\[\[\[1. 1. 1.\][[:space:]]*\[1. 1. 1.\]\]\]}}
# CHECK-NEXT: -- Memref shape: [1, 2, 3]
# CHECK-NEXT: -- Memref dtype: ScalarTypeCode.f32
# CHECK-NEXT: -- numpy.from_dlpack(): {{\[\[\[1. 1. 1.\][[:space:]]*\[1. 1. 1.\]\]\]}}
# CHECK-NEXT: Array: []
# CHECK-NEXT: -- Memref shape: [0]
# CHECK-NEXT: -- Memref dtype: ScalarTypeCode.i8
# CHECK-NEXT: -- numpy.from_dlpack(): []
# CHECK-LABEL: Test cp.array -> client.create_memref_from_dlpack
# CHECK-NEXT: Array: [1 2 3 4]
# CHECK-NEXT: -- Memref shape: [4]
# CHECK-NEXT: -- Memref dtype: ScalarTypeCode.i32
# CHECK-NEXT: -- cupy.from_dlpack(): [1 2 3 4]
# CHECK-NEXT: Array: {{\[\[\[1. 1. 1.\][[:space:]]*\[1. 1. 1.\]\]\]}}
# CHECK-NEXT: -- Memref shape: [1, 2, 3]
# CHECK-NEXT: -- Memref dtype: ScalarTypeCode.f32
# CHECK-NEXT: -- cupy.from_dlpack(): {{\[\[\[1. 1. 1.\][[:space:]]*\[1. 1. 1.\]\]\]}}
# CHECK-NEXT: Array: []
# CHECK-NEXT: -- Memref shape: [0]
# CHECK-NEXT: -- Memref dtype: ScalarTypeCode.i8
# CHECK-NEXT: -- cupy.from_dlpack(): []


def test_client_destroyed_before_external_dlpack(dl):
    new_client = runtime.RuntimeClient()

    def create_memref_dlpackview_from_dlpack(dl, module):
        memref = new_client.create_memref_view_from_dlpack(dl)
        dl = module.from_dlpack(memref)
        print(f"-- Memref shape: {memref.shape}")
        print(f"-- Memref dtype: {memref.dtype}")
        print(f"-- {module.__name__}.from_dlpack(): {dl}")
        return memref, dl

    memref1, dl1 = create_memref_dlpackview_from_dlpack(dl, np)
    memref2, dl2 = create_memref_dlpackview_from_dlpack(dl, np)
    del new_client
    gc.collect()
    return memref1, dl1, memref2, dl2


print("Test keeping np.array, dlpack, memref view and dlpack view alive.")
arr = np.array([1, 2, 3, 4], dtype=np.int32)
m1, d1, m2, d2 = test_client_destroyed_before_external_dlpack(arr.__dlpack__())

# CHECK-LABEL: Test keeping np.array, dlpack, memref view and dlpack view alive.
# CHECK-NEXT: -- Memref shape: [4]
# CHECK-NEXT: -- Memref dtype: ScalarTypeCode.i32
# CHECK-NEXT: -- numpy.from_dlpack(): [1 2 3 4]
# CHECK-NEXT: -- Memref shape: [4]
# CHECK-NEXT: -- Memref dtype: ScalarTypeCode.i32


def create_dangling_memref():
    array = np.array([1, 2])
    dlpack_capsule = array.__dlpack__()
    memref = client.create_memref_view_from_dlpack(dlpack_capsule)
    print("-- Inner scope: np.from_dlpack(): ", np.from_dlpack(memref))
    return memref


print("Test memref maintains data's lifetime")
memref = create_dangling_memref()
# Declare a new array to overwrite the old memory
b = np.array([10, 10])
print("-- Outer scope: np.from_dlpack(): ", np.from_dlpack(memref))


# CHECK-LABEL: Test memref maintains data's lifetime
# CHECK-NEXT: -- Inner scope: np.from_dlpack(): [1 2]
# CHECK-NEXT: -- Outer scope: np.from_dlpack(): [1 2]


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
