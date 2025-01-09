from __future__ import annotations

import typing

import numpy as np
import typing_extensions

__all__ = [
    "Device",
    "Executable",
    "MTRTException",
    "MemRefType",
    "MemRefValue",
    "PointerType",
    "PyBounds",
    "PyFunctionSignature",
    "RuntimeClient",
    "RuntimeSession",
    "RuntimeSessionOptions",
    "RuntimeValue",
    "ScalarType",
    "ScalarTypeCode",
    "ScalarValue",
    "Stream",
    "Type",
    "bf16",
    "device",
    "f16",
    "f32",
    "f64",
    "f8e4m3fn",
    "host",
    "i1",
    "i16",
    "i32",
    "i64",
    "i8",
    "pinned_host",
    "ui8",
    "unified",
    "unknown",
]

class Device:
    @property
    def _CAPIPtr(self) -> typing.Any: ...

class Executable:
    def __init__(self, buffer: bytes) -> None: ...
    def get_signature(self, arg0: str) -> PyFunctionSignature: ...
    def serialize(self) -> bytes: ...

class MTRTException(Exception):
    pass

class MemRefType(Type):
    @staticmethod
    def get(
        shape: list[int], elementType: ScalarTypeCode, addressSpace: PointerType
    ) -> MemRefType:
        """
        construct a memref type
        """

    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str:
        """
        returns string representation of memref type
        """

    @property
    def _CAPIPtr(self) -> typing.Any: ...
    @property
    def address_space(self) -> PointerType: ...
    @property
    def dtype(self) -> ScalarTypeCode: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def strides(self) -> list[int]: ...

class MemRefValue:
    def __dlpack__(self, stream: int = 0) -> object: ...
    def __dlpack_device__(self) -> tuple: ...
    @property
    def _CAPIPtr(self) -> typing.Any: ...
    @property
    def address_space(self) -> PointerType: ...
    @property
    def dtype(self) -> ScalarTypeCode: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def strides(self) -> list[int]: ...

class PointerType:
    """
    Members:

      host

      pinned_host

      device

      unified

      unknown
    """

    __members__: typing.ClassVar[
        dict[str, PointerType]
    ]  # value = {'host': <PointerType.host: 0>, 'pinned_host': <PointerType.pinned_host: 1>, 'device': <PointerType.device: 2>, 'unified': <PointerType.unified: 3>, 'unknown': <PointerType.unknown: 4>}
    device: typing.ClassVar[PointerType]  # value = <PointerType.device: 2>
    host: typing.ClassVar[PointerType]  # value = <PointerType.host: 0>
    pinned_host: typing.ClassVar[PointerType]  # value = <PointerType.pinned_host: 1>
    unified: typing.ClassVar[PointerType]  # value = <PointerType.unified: 3>
    unknown: typing.ClassVar[PointerType]  # value = <PointerType.unknown: 4>
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class PyBounds:
    def max(self) -> list[int]: ...
    def min(self) -> list[int]: ...

class PyFunctionSignature:
    def __str__(self) -> str:
        """
        returns string representation of function signature type
        """

    def get_arg(self, arg0: int) -> Type: ...
    def get_arg_bound(self, arg0: int) -> PyBounds: ...
    def get_num_arg_bounds(self) -> int: ...
    def get_num_args(self) -> int: ...
    def get_num_input_args(self) -> int: ...
    def get_num_output_args(self) -> int: ...
    def get_num_res_bounds(self) -> int: ...
    def get_num_results(self) -> int: ...
    def get_res_bound(self, arg0: int) -> PyBounds: ...
    def get_result(self, arg0: int) -> Type: ...
    def get_shape_func_name(self) -> str | None:
        """
        returns the name of the MLIR-TensorRT function in the same executable that computes the result shapes from the input shapes if available, otherwise it runs None
        """

class RuntimeClient:
    def __init__(self) -> None: ...
    def copy_to_device(
        self,
        host_memref: MemRefValue,
        device: Device,
        stream: Stream | None = None,
    ) -> MemRefValue: ...
    @typing.overload
    def copy_to_host(
        self, device_memref: MemRefValue, stream: Stream | None = None
    ) -> MemRefValue: ...
    @typing.overload
    def copy_to_host(
        self,
        device_memref: MemRefValue,
        existing_host_memref: MemRefValue,
        stream: Stream | None = None,
    ) -> None: ...
    def create_device_memref_view(
        self, ptr: int, shape: list[int], dtype: ScalarTypeCode, device: Device
    ) -> MemRefValue: ...
    def create_host_memref_view(
        self, ptr: int, shape: list[int], dtype: ScalarTypeCode
    ) -> MemRefValue: ...
    @typing.overload
    def create_memref(
        self,
        shape: list[int],
        dtype: ScalarTypeCode,
        device: Device | None = None,
        stream: Stream | None = None,
    ) -> MemRefValue: ...
    @typing.overload
    def create_memref(
        self,
        array: typing_extensions.Buffer | np.ndarray,
        shape: list[int] | None = None,
        dtype: ScalarTypeCode | None = None,
        device: Device | None = None,
        stream: Stream | None = None,
    ) -> MemRefValue: ...
    def create_scalar(
        self, scalar_value: typing.Any, type_code: ScalarTypeCode | None
    ) -> ScalarValue:
        """
        creates a runtime ScalarValue from the provided Python object; an explicit type may be provided, otherwise defaults to i64 for Python integers and f32 for Python floats
        """

    def create_stream(self) -> Stream: ...
    def get_devices(self) -> list[Device]: ...

class RuntimeSession:
    def __init__(
        self, options: RuntimeSessionOptions, executable: Executable
    ) -> None: ...
    def execute_function(
        self,
        name: str,
        in_args: list[typing.Any],
        out_args: list[typing.Any],
        stream: Stream | None = None,
    ) -> None: ...

class RuntimeSessionOptions:
    def __init__(
        self, num_devices: int = 1, device_id: int = 0, nccl_uuid: str = ""
    ) -> None: ...

class RuntimeValue:
    @typing.overload
    def __init__(self, scalar_int: int) -> None: ...
    @typing.overload
    def __init__(
        self, pointer: int, offset: int, shape: list[int], strides: list[int]
    ) -> None: ...
    @property
    def _CAPIPtr(self) -> typing.Any: ...

class ScalarType(Type):
    @staticmethod
    def get(type_code: ScalarTypeCode) -> ScalarType: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str:
        """
        returns string representation of the type
        """

class ScalarTypeCode:
    """
    Members:

      f8e4m3fn

      f16

      bf16

      f32

      f64

      i1

      i8

      ui8

      i16

      i32

      i64
    """

    __members__: typing.ClassVar[
        dict[str, ScalarTypeCode]
    ]  # value = {'f8e4m3fn': <ScalarTypeCode.f8e4m3fn: 1>, 'f16': <ScalarTypeCode.f16: 2>, 'bf16': <ScalarTypeCode.bf16: 11>, 'f32': <ScalarTypeCode.f32: 3>, 'f64': <ScalarTypeCode.f64: 4>, 'i1': <ScalarTypeCode.i1: 5>, 'i8': <ScalarTypeCode.i8: 6>, 'ui8': <ScalarTypeCode.ui8: 7>, 'i16': <ScalarTypeCode.i16: 8>, 'i32': <ScalarTypeCode.i32: 9>, 'i64': <ScalarTypeCode.i64: 10>}
    bf16: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.bf16: 11>
    f16: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.f16: 2>
    f32: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.f32: 3>
    f64: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.f64: 4>
    f8e4m3fn: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.f8e4m3fn: 1>
    i1: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.i1: 5>
    i16: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.i16: 8>
    i32: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.i32: 9>
    i64: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.i64: 10>
    i8: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.i8: 6>
    ui8: typing.ClassVar[ScalarTypeCode]  # value = <ScalarTypeCode.ui8: 7>
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ScalarValue:
    @property
    def _CAPIPtr(self) -> typing.Any: ...
    @property
    def type(self) -> ScalarTypeCode: ...

class Stream:
    def sync(self) -> None: ...
    @property
    def _CAPIPtr(self) -> typing.Any: ...

class Type:
    def __init__(self, cast_from_type: Type) -> None: ...

bf16: ScalarTypeCode  # value = <ScalarTypeCode.bf16: 11>
device: PointerType  # value = <PointerType.device: 2>
f16: ScalarTypeCode  # value = <ScalarTypeCode.f16: 2>
f32: ScalarTypeCode  # value = <ScalarTypeCode.f32: 3>
f64: ScalarTypeCode  # value = <ScalarTypeCode.f64: 4>
f8e4m3fn: ScalarTypeCode  # value = <ScalarTypeCode.f8e4m3fn: 1>
host: PointerType  # value = <PointerType.host: 0>
i1: ScalarTypeCode  # value = <ScalarTypeCode.i1: 5>
i16: ScalarTypeCode  # value = <ScalarTypeCode.i16: 8>
i32: ScalarTypeCode  # value = <ScalarTypeCode.i32: 9>
i64: ScalarTypeCode  # value = <ScalarTypeCode.i64: 10>
i8: ScalarTypeCode  # value = <ScalarTypeCode.i8: 6>
pinned_host: PointerType  # value = <PointerType.pinned_host: 1>
ui8: ScalarTypeCode  # value = <ScalarTypeCode.ui8: 7>
unified: PointerType  # value = <PointerType.unified: 3>
unknown: PointerType  # value = <PointerType.unknown: 4>
