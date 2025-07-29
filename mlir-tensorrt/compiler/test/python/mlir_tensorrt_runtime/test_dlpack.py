# RUN: %PYTHON %s
# Tests DLPack <-> MemRefValue conversions.
import array
import gc
import os
import sys

import mlir_tensorrt.runtime.api as runtime
import numpy as np

client = runtime.RuntimeClient()
stream = client.create_stream()
devices = client.get_devices()


def check_host_memref_to_np_dlpack():
    """This is a basic sanity check MemRefValue -> np.array via DLPack"""
    arr = np.array([5.0, 4.0, 2.0])
    memref = client.create_host_memref_view(
        int(arr.ctypes.data), shape=[3], dtype=runtime.ScalarTypeCode.f64
    )
    assert memref.ref_count() == 1, "incorrect initial ref count"

    # Create DLPack from Python
    np_array = np.from_dlpack(memref)
    assert memref.ref_count() == 2, "incorrect ref count after creating dlpack"

    del np_array
    gc.collect()
    assert memref.ref_count() == 1, "incorrect ref count after deleting dlpack"

    # Now releasing the memref shouldn't try to deallocate anything. Trying to
    # deallocate will crash or throw an exception depending on what level of runtime
    # debug checks are enabled.
    del memref
    gc.collect()


check_host_memref_to_np_dlpack()


def test_client_destroyed_before_external_dlpack(data):
    """Create a MemRefValue from a DLPack, and then delete the MTRT objects in
    Python while the DLPack is still live. This checks that the Client should
    remain alive in C++ until the last reference to any MemRefValue is dropped."""

    new_client = runtime.RuntimeClient()

    def create_memref_dlpackview_from_dlpack():
        arr_in = array.array("f", [5.0, 4.0, 2.0])
        memref = new_client.create_memref(arr_in)
        dl = np.from_dlpack(memref)
        return dl

    dl1 = create_memref_dlpackview_from_dlpack()
    dl2 = create_memref_dlpackview_from_dlpack()
    del new_client
    gc.collect()
    del dl1, dl2
    gc.collect()


test_client_destroyed_before_external_dlpack([1, 2, 3, 4])


def roundtrip_dlpack_from_mtrt():
    """Create a MemRefValue, then convert to DLPack and back to another MemRefValue.
    This checks that deleting the original MemRefValue will not free the underlying
    storage until the reference round-tripped through dlpack is also dropped.
    """
    arr_in = array.array("f", [5.0, 4.0, 2.0])
    memref1 = client.create_memref(arr_in)
    dlpack = memref1.__dlpack__()
    memref2 = client.create_memref_view_from_dlpack(dlpack)

    assert memref1.ref_count() == 2, "incorrect ref count"
    # memref2's storage is view storage. It has its own reference count. The second reference
    # to memref1's storage lives in the dlpack manager's context, which isn't
    # released until memref2 is released.
    assert memref2.ref_count() == 1, "incorrect ref count"

    del memref1
    gc.collect()
    assert memref2.ref_count() == 1, "incorrect ref count"


roundtrip_dlpack_from_mtrt()


def roundtrip_dlpack_from_mtrt_2():
    """Create a MemRefValue, then convert to DLPack and back to another MemRefValue.
    This checks that deleting the second MemRefValue will correctly drop the
    reference to the first memref value.
    """
    arr_in = array.array("f", [5.0, 4.0, 2.0])
    memref1 = client.create_memref(arr_in)
    memref_views = [
        client.create_memref_view_from_dlpack(memref1.__dlpack__()) for i in range(10)
    ]
    assert all(
        memref.ref_count() == 1 for memref in memref_views
    ), "incorrect ref count"

    assert memref1.ref_count() == 11, "incorrect ref count"

    del memref_views
    gc.collect()

    assert memref1.ref_count() == 1, "incorrect ref count"


roundtrip_dlpack_from_mtrt_2()


def check_dlpack_source_lifetime():
    """Basic sanity check for checking that DLPack lifetime is properly
    matched to the result of 'create_memref_view_from_dlpack' via
    callback mechanism.
    """
    array = np.array([1, 2])
    dlpack_capsule = array.__dlpack__()
    memref = client.create_memref_view_from_dlpack(dlpack_capsule)
    del array
    gc.collect()
    np.testing.assert_array_equal(np.from_dlpack(memref), np.array([1, 2]))


check_dlpack_source_lifetime()
