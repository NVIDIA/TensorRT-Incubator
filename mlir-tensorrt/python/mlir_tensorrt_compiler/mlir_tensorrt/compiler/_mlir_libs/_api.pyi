from __future__ import annotations
import typing
from mlir_tensorrt.compiler.ir import Context, Operation, FunctionType

__all__ = [
    "CompilerClient",
    "Executable",
    "MemRefType",
    "PluginFieldInfo",
    "PluginFieldType",
    "PointerType",
    "PyBounds",
    "PyFunctionSignature",
    "ScalarType",
    "ScalarTypeCode",
    "StableHLOToExecutableOptions",
    "Type",
    "bf16",
    "compiler_stablehlo_to_executable",
    "device",
    "f16",
    "f32",
    "f64",
    "f8e4m3fn",
    "get_stablehlo_program_refined_signature",
    "get_tensorrt_plugin_field_schema",
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

class CompilerClient:
    def __init__(self, arg0: Context) -> None: ...

class Executable:
    def __init__(self, buffer: str) -> None:
        """
        constructs an executable from a bytes buffer.
        """

    def get_signature(self, arg0: str) -> ...: ...
    def serialize(self) -> bytes | None:
        """
        returns serialized executable in `bytes`
        """

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

class PluginFieldInfo:
    @property
    def length(self) -> int: ...
    @property
    def type(self) -> PluginFieldType: ...

class PluginFieldType:
    """
    Members:

      FLOAT16

      FLOAT32

      FLOAT64

      INT8

      INT16

      INT32

      CHAR

      DIMS

      UNKNOWN

      BF16

      INT64

      FP8

      INT4
    """

    BF16: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.BF16: 9>
    CHAR: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.CHAR: 6>
    DIMS: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.DIMS: 7>
    FLOAT16: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.FLOAT16: 0>
    FLOAT32: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.FLOAT32: 1>
    FLOAT64: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.FLOAT64: 2>
    FP8: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.FP8: 11>
    INT16: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.INT16: 4>
    INT32: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.INT32: 5>
    INT4: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.INT4: 12>
    INT64: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.INT64: 10>
    INT8: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.INT8: 3>
    UNKNOWN: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.UNKNOWN: 8>
    __members__: typing.ClassVar[
        dict[str, PluginFieldType]
    ]  # value = {'FLOAT16': <PluginFieldType.FLOAT16: 0>, 'FLOAT32': <PluginFieldType.FLOAT32: 1>, 'FLOAT64': <PluginFieldType.FLOAT64: 2>, 'INT8': <PluginFieldType.INT8: 3>, 'INT16': <PluginFieldType.INT16: 4>, 'INT32': <PluginFieldType.INT32: 5>, 'CHAR': <PluginFieldType.CHAR: 6>, 'DIMS': <PluginFieldType.DIMS: 7>, 'UNKNOWN': <PluginFieldType.UNKNOWN: 8>, 'BF16': <PluginFieldType.BF16: 9>, 'INT64': <PluginFieldType.INT64: 10>, 'FP8': <PluginFieldType.FP8: 11>, 'INT4': <PluginFieldType.INT4: 12>}
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

class StableHLOToExecutableOptions:
    def __init__(
        self, tensorrt_builder_opt_level: int, tensorrt_strongly_typed: bool
    ) -> None: ...
    def set_debug_options(
        self,
        enabled: bool,
        debug_types: list[str] = [],
        dump_ir_tree_dir: str | None = None,
        dump_tensorrt_dir: str | None = None,
    ) -> None: ...

class Type:
    def __init__(self, cast_from_type: Type) -> None: ...

def compiler_stablehlo_to_executable(
    client: CompilerClient, module: Operation, options: StableHLOToExecutableOptions
) -> Executable: ...
def get_stablehlo_program_refined_signature(
    client: CompilerClient, module: Operation, func_name: str
) -> FunctionType: ...
def get_tensorrt_plugin_field_schema(
    name: str, version: str, plugin_namespace: str, dso_path: str
) -> dict[str, PluginFieldInfo]:
    """
    Queries the global TensorRT plugin registry for a creator for a plugin of the given name, version, and namespace. It then queries the plugin creator for the expected PluginField information.
    """

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
