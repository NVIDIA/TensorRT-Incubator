import abc
from typing import List
from tripy.common.types import ShapeInfo
from tripy.common.datatype import dtype
from dataclasses import dataclass


@dataclass
class BaseOperator(abc.ABC):
    inputs: List["TraceTensor"]
    """The inputs of this layer"""

    outputs: List["TraceTensor"]
    """The outputs of this layer"""

    const_fold: bool
    """Whether to treat the operation as a constant in JIT"""

    @abc.abstractmethod
    def to_trace_str(self) -> str:
        """
        Returns a Trace string representation of the operation.

        Returns:
            The Trace string representation of the operation.
        """
        ...

    @abc.abstractmethod
    def infer_shapes(self):
        """
        Infers shapes for the operation and updates output tensor shapes accordingly.
        """
        ...

    def infer_dtypes(self):
        """
        Infers dtypes for the operation and updates output tensor dtypes accordingly.
        """
        assert (
            self.inputs and len(self.outputs) == 1 and all(inp.dtype == self.inputs[0].dtype for inp in self.inputs)
        ), "Default implementation cannot handle cases where there are no inputs, multiple outputs, or multiple inputs with different data types. Please override."
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_devices(self):
        """
        Infers devices for the operation and updates output tensor devices accordingly.
        """
        assert (
            self.inputs and len(self.outputs) == 1 and all(inp.device == self.inputs[0].device for inp in self.inputs)
        ), "Default implementation cannot handle cases where there are no inputs, multiple outputs, or multiple inputs with different devices. Please override."
        self.outputs[0].device = self.inputs[0].device

    @abc.abstractmethod
    def to_flat_ir(self, flat_ir) -> None:
        """
        Generates FlatIR ops for the operation.

        Args:
            flat_ir: FlatIR parent graph where new ops are inserted.
        """
        ...
