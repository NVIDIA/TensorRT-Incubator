# REQUIRES: host-has-at-least-1-gpus
# RUN: %pick-one-gpu %PYTHON %s | FileCheck %s

from typing import Callable

import mlir_tensorrt.runtime.api as runtime
import numpy as np

TESTS = []


def make_test(callable: Callable[[], None]):
    def _test():
        print("Test: ", callable.__name__)
        callable()
        print("\n ----- \n")

    TESTS.append(_test)


@make_test
def test_enums():
    def _print_enum(type):
        if not hasattr(type, "__members__"):
            raise TypeError(
                f"The type {type.__name__} does not have a '__members__' attribute."
            )
        for name, value in type.__members__.items():
            print(name, value, value.value)

    _print_enum(runtime.ScalarTypeCode)
    _print_enum(runtime.PointerType)


# CHECK-LABEL: Test:  test_enums
#       CHECK:   f8e4m3fn ScalarTypeCode.f8e4m3fn 1
#       CHECK:   f16 ScalarTypeCode.f16 2
#       CHECK:   bf16 ScalarTypeCode.bf16 11
#       CHECK:   f32 ScalarTypeCode.f32 3
#       CHECK:   f64 ScalarTypeCode.f64 4
#       CHECK:   i1 ScalarTypeCode.i1 5
#       CHECK:   i8 ScalarTypeCode.i8 6
#       CHECK:   ui8 ScalarTypeCode.ui8 7
#       CHECK:   i16 ScalarTypeCode.i16 8
#       CHECK:   i32 ScalarTypeCode.i32 9
#       CHECK:   i64 ScalarTypeCode.i64 10
#       CHECK:   complex32 ScalarTypeCode.complex32 13
#       CHECK:   complex64 ScalarTypeCode.complex64 14
#       CHECK:   f4e2m1fn ScalarTypeCode.f4e2m1fn 15
#       CHECK:   host PointerType.host 0
#       CHECK:   pinned_host PointerType.pinned_host 1
#       CHECK:   device PointerType.device 2
#       CHECK:   unified PointerType.unified 3
#       CHECK:   unknown PointerType.unknown 4


@make_test
def test_memref():
    client = runtime.RuntimeClient()
    devices = client.get_devices()
    if len(devices) == 0:
        return

    # Test numpy -> device -> numpy roundtrip for all data types
    # supported by MTRT and buffer protocol.
    # We don't support F64, so test that the approriate error is raised.
    for dtype in [
        np.float64,
        np.float32,
        np.float16,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
    ]:
        try:
            print(f"testing dtype {dtype.__name__}")
            src_data = np.arange(0, 16, dtype=dtype).reshape(2, 8)
            gpu_array = client.create_memref(src_data, device=devices[0])
            print(gpu_array.shape)
            print(gpu_array.strides)
            print(gpu_array.address_space)
            print(np.asarray(client.copy_to_host(gpu_array)))
            print(np.from_dlpack(client.copy_to_host(gpu_array)))
        except Exception as e:
            print("Exception caught: ", e)


# CHECK-LABEL: Test: test_memref
#  CHECK-NEXT: testing dtype float64
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.device
#  CHECK-NEXT:  [ 0.  1.  2.  3.  4.  5.  6.  7.]
#  CHECK-NEXT:  [ 8.  9. 10. 11. 12. 13. 14. 15.]]
#  CHECK-NEXT:  [ 0.  1.  2.  3.  4.  5.  6.  7.]
#  CHECK-NEXT:  [ 8.  9. 10. 11. 12. 13. 14. 15.]]
#  CHECK-NEXT: testing dtype float32
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.device
#  CHECK-NEXT:  [ 0.  1.  2.  3.  4.  5.  6.  7.]
#  CHECK-NEXT:  [ 8.  9. 10. 11. 12. 13. 14. 15.]]
#  CHECK-NEXT:  [ 0.  1.  2.  3.  4.  5.  6.  7.]
#  CHECK-NEXT:  [ 8.  9. 10. 11. 12. 13. 14. 15.]]
#  CHECK-NEXT: testing dtype float16
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.device
#  CHECK-NEXT:  [ 0.  1.  2.  3.  4.  5.  6.  7.]
#  CHECK-NEXT:  [ 8.  9. 10. 11. 12. 13. 14. 15.]]
#  CHECK-NEXT:  [ 0.  1.  2.  3.  4.  5.  6.  7.]
#  CHECK-NEXT:  [ 8.  9. 10. 11. 12. 13. 14. 15.]]
#  CHECK-NEXT: testing dtype int64
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.device
#  CHECK-NEXT:  [ 0  1  2  3  4  5  6  7]
#  CHECK-NEXT:  [ 8  9 10 11 12 13 14 15]]
#  CHECK-NEXT:  [ 0  1  2  3  4  5  6  7]
#  CHECK-NEXT:  [ 8  9 10 11 12 13 14 15]]
#  CHECK-NEXT: testing dtype int32
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.device
#  CHECK-NEXT:  [ 0  1  2  3  4  5  6  7]
#  CHECK-NEXT:  [ 8  9 10 11 12 13 14 15]]
#  CHECK-NEXT:  [ 0  1  2  3  4  5  6  7]
#  CHECK-NEXT:  [ 8  9 10 11 12 13 14 15]]
#  CHECK-NEXT: testing dtype int16
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.device
#  CHECK-NEXT:  [ 0  1  2  3  4  5  6  7]
#  CHECK-NEXT:  [ 8  9 10 11 12 13 14 15]]
#  CHECK-NEXT:  [ 0  1  2  3  4  5  6  7]
#  CHECK-NEXT:  [ 8  9 10 11 12 13 14 15]]
#  CHECK-NEXT: testing dtype int8
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.device
#  CHECK-NEXT:  [ 0  1  2  3  4  5  6  7]
#  CHECK-NEXT:  [ 8  9 10 11 12 13 14 15]]
#  CHECK-NEXT:  [ 0  1  2  3  4  5  6  7]
#  CHECK-NEXT:  [ 8  9 10 11 12 13 14 15]]


@make_test
def test_host_memref():
    client = runtime.RuntimeClient()

    # Test host memref creation from an external host data for all data types
    # supported by MTRT and buffer protocol.
    # We don't support F64, so test that the appropriate error is raised.
    for dtype in [
        np.float64,
        np.float32,
        np.float16,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
    ]:
        try:
            print(f"testing dtype {dtype.__name__}")
            src_data = np.arange(0, 16, dtype=dtype).reshape(2, 8)
            host_array = client.create_memref(src_data)
            print(host_array.shape)
            print(host_array.strides)
            print(host_array.address_space)
        except Exception as e:
            print("Exception caught: ", e)


# CHECK-LABEL: Test: test_host_memref
#  CHECK-NEXT: testing dtype float64
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.host
#  CHECK-NEXT: testing dtype float32
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.host
#  CHECK-NEXT: testing dtype float16
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.host
#  CHECK-NEXT: testing dtype int64
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.host
#  CHECK-NEXT: testing dtype int32
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.host
#  CHECK-NEXT: testing dtype int16
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.host
#  CHECK-NEXT: testing dtype int8
#  CHECK-NEXT: [2, 8]
#  CHECK-NEXT: [8, 1]
#  CHECK-NEXT: PointerType.host


@make_test
def test_devices():
    client = runtime.RuntimeClient()
    devices = client.get_devices()
    if len(devices) == 0:
        return
    try:
        print("Device name:", devices[0].get_name())
    except Exception as e:
        print("Exception caught: ", e)


# CHECK-LABEL: Test: test_devices
#  CHECK-NEXT: Device name: cuda:0


@make_test
def test_stream():
    client = runtime.RuntimeClient()
    devices = client.get_devices()
    if len(devices) == 0:
        return
    stream = devices[0].stream
    assert isinstance(stream.ptr, int)


# CHECK-LABEL: Test: test_stream


if __name__ == "__main__":
    for t in TESTS:
        t()
