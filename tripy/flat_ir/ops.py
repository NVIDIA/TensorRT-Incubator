import abc
from dataclasses import dataclass
from typing import List

from mlir import ir
from mlir.dialects import stablehlo


@dataclass
class FIROps(abc.ABC):
    """
    Represents a single layer in the FlatIR.
    """

    inputs: List["FIRTensor"]
    """The inputs of this layer"""

    outputs: List["FIRTensor"]
    """The outputs of this layer"""

    origin_layer: "TraceLayer"
    """The operation applied by this layer"""

    @abc.abstractmethod
    def to_mlir(self, inputs: List) -> List:
        """
        Generates MLIR HLO ops for the operation.

        Args:
            inputs: The input MLIR HLO operations.

        Returns:
            The MLIR HLO op(s) corresponding to this operation.
        """
        ...

    @abc.abstractmethod
    def to_flat_ir_str(self, inputs: List, output_names: List) -> str:
        """
        Returns a FlatIR string representation of the operation.

        Args:
            inputs_names: The names of the input tensor(s).
            output_names: The names of the output tensor(s).

        Returns:
            The FlatIR string representation of the operation.
        """
        ...


class AddOp(FIROps):
    """
    Operation to add two tensors
    """

    def __init__(self, origin_layer, inputs, outputs):
        super().__init__(inputs, outputs, origin_layer)

    def to_flat_ir_str(self, input_names, output_names) -> str:
        assert len(output_names) == 1, "AddOp should have exactly one output!"
        return f"{output_names[0]} = {self.__class__.__name__} {' '.join(input_names)}"

    def to_mlir(self, operands: List) -> List:
        add_out = stablehlo.AddOp(*operands)
        return [add_out]


class CompareOp(FIROps):
    """
    Operation to compare two tensors
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert "compare_direction" in kwargs
        self.compare_direction = kwargs.get("compare_direction")

    def add_spaces_around_string(self, s):
        return f" {s} "

    def to_flat_ir_str(self, input_names, output_names) -> str:
        assert len(output_names) == 1, "CompareOp should have exactly one output!"
        return f"{output_names[0]} = {self.__class__.__name__}.{self.compare_direction} {' '.join(input_names)}"

    def to_mlir(self, operands: List) -> List:
        compare_out = stablehlo.CompareOp(*operands, stablehlo.ComparisonDirectionAttr.get(self.compare_direction))
        return [compare_out]


class ConstantOp(FIROps):
    """
    Operation to store a constant
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert "data" in kwargs
        self.data = kwargs.get("data")

    def to_flat_ir_str(self, input_names, output_names) -> str:
        data = self.origin_layer.data.view()
        return f"{output_names[0]} : {self.__class__.__name__} data=({data.view()}), shape=({data.shape}), dtype=({self.origin_layer.dtype.name}), loc=({self.origin_layer.device.kind}:{self.origin_layer.device.index})"

    def to_mlir(self, operands):
        from tripy.backend.mlir import utils as mlir_utils

        attr = ir.DenseElementsAttr.get(
            array=self.data, type=mlir_utils.get_mlir_dtype(self.origin_layer.dtype), shape=self.data.shape
        )
        return [stablehlo.ConstantOp(attr)]


class CopyOp(FIROps):
    """
    Operation to copy a tensor to another device
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert "target" in kwargs
        self.target = kwargs.get("target")

    def to_flat_ir_str(self, input_names, output_names) -> str:
        assert len(output_names) == 1, "CompareOp should have exactly one output!"
        return f"{output_names[0]} : {self.__class__.__name__} copy={input_names[0]}, target={self.target.kind}:{self.target.index})"

    def to_mlir(self, operands: List) -> List:
        from mlir.dialects import bufferization

        assert len(operands) == 1 and len(self.inputs) == 1, "Copy should have exactly one input!"
        mem_space_str = "device" if self.target.kind == "gpu" else "host_pinned"
        mem_space_attr = ir.Attribute.parse(f"#executor.memory_type<{mem_space_str}>")
        dst_tensor = bufferization.alloc_tensor(
            self.inputs[0].to_mlir(), [], memory_space=mem_space_attr, copy=operands[0]
        )
        return [dst_tensor]
