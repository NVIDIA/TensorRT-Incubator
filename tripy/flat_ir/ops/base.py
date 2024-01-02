import abc
from dataclasses import dataclass
from typing import List


@dataclass
class BaseFIROp(abc.ABC):
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
