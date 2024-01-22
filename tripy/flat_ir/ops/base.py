import abc
import dataclasses
from dataclasses import dataclass
from typing import List


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
    def to_mlir(self, inputs: List) -> List:
        """
        Generates MLIR HLO ops for the operation.

        Args:
            inputs: The input MLIR HLO operations.

        Returns:
            The MLIR HLO op(s) corresponding to this operation.
        """
        ...

    def __str__(self) -> str:
        """
        Returns a FlatIR string representation of the operation.

        Returns:
            The FlatIR string representation of the operation.
        """
        outputs_str = (
            f"{str(self.outputs[0])}" if len(self.outputs) == 1 else ", ".join([str(out) for out in self.outputs])
        )
        args = [
            f"{field.name}={getattr(self, field.name)}"
            for field in dataclasses.fields(self)
            if field.name not in [base_field.name for base_field in dataclasses.fields(BaseFIROp)]
        ]
        return f"{outputs_str} = {self.name()}({', '.join([inp.name for inp in self.inputs] + args)})"

    def name(self) -> str:
        """
        Returns the human readable name of this operation.

        Returns:
            The name of this operation.
        """
        return self.__class__.__name__
