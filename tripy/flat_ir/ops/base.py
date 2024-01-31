import abc
from dataclasses import dataclass
from typing import List, Set

from tripy import utils


@dataclass
class BaseFIROp(abc.ABC):
    """
    Represents a single layer in the FlatIR.
    """

    origin_layer: "BaseOperator"
    """The frontend operator that generated this op"""

    inputs: List["FIRTensor"]
    """The inputs of this layer"""

    outputs: List["FIRTensor"]
    """The outputs of this layer"""

    def __init__(self, origin_layer: "BaseOperator", inputs: List["FIRTensor"], outputs: List["FIRTensor"]):
        from tripy.flat_ir.tensor import FIRTensor

        assert all(isinstance(tensor, FIRTensor) for tensor in inputs + outputs)

        self.inputs = inputs
        self.outputs = outputs
        self.origin_layer = origin_layer

        for out in self.outputs:
            out.producer = self

    @abc.abstractmethod
    def to_mlir(self, operands: List["ir.Operation"]) -> List["ir.Operation"]:
        """
        Generates MLIR HLO operations for the operation.

        Args:
            operands: The input MLIR HLO operations.

        Returns:
            The generated MLIR HLO operations.
        """
        ...

    def str_skip_fields(self) -> Set[str]:
        """
        Returns names of dataclass fields to skip when generating a string representation of the op.
        """
        return set()

    def __str__(self) -> str:
        """
        Returns a FlatIR string representation of the operation.

        Returns:
            The FlatIR string representation of the operation.
        """
        outputs_str = (
            f"{str(self.outputs[0])}" if len(self.outputs) == 1 else ", ".join([str(out) for out in self.outputs])
        )
        skip_fields = self.str_skip_fields()
        args = [
            f"{field.name}={getattr(self, field.name)}"
            for field in utils.get_dataclass_fields(self, BaseFIROp)
            if field.name not in skip_fields
        ]
        return f"{outputs_str} = {self.name()}({', '.join([inp.name for inp in self.inputs] + args)})"

    def name(self) -> str:
        """
        Returns the human readable name of this operation.

        Returns:
            The name of this operation.
        """
        return self.__class__.__name__
