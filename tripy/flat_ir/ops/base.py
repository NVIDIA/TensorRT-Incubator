import abc
from dataclasses import dataclass
from typing import List


@dataclass
class BaseFIROp(abc.ABC):
    """
    Represents a single layer in the FlatIR.
    """

    origin_layer: "TraceLayer"
    """The trace layer used to generate this layer"""

    inputs: List["FIRTensor"]
    """The inputs of this layer"""

    outputs: List["FIRTensor"]
    """The outputs of this layer"""

    def __init__(self, origin_layer: "TraceLayer", inputs: List["TraceTensor"], outputs: List["TraceTensor"]):
        from tripy.flat_ir.tensor import FIRTensor

        self.inputs = list(map(FIRTensor, inputs))
        self.outputs = list(map(FIRTensor, outputs))
        self.origin_layer = origin_layer

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

    def to_flat_ir_str(self) -> str:
        """
        Returns a FlatIR string representation of the operation.

        Returns:
            The FlatIR string representation of the operation.
        """
        outputs_str = f"{self.outputs[0].name}" if len(self.outputs) == 1 else str([out.name for out in self.outputs])

        return f"{outputs_str} = {self.name()}({', '.join(list(map(str, self.inputs)))})"

    def name(self) -> str:
        """
        Returns the human readable name of this operation.

        Returns:
            The name of this operation.
        """
        return self.__class__.__name__
