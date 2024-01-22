import abc
import dataclasses
from dataclasses import dataclass
from typing import List


@dataclass
class BaseOperator(abc.ABC):
    inputs: List["TraceTensor"]
    """The inputs of this layer"""

    outputs: List["TraceTensor"]
    """The outputs of this layer"""

    const_fold: bool
    """Whether to treat the operation as a constant in JIT"""

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
    def to_flat_ir(self, inputs: List["FIRTensor"], outputs: List["FIRTensor"]):
        """
        Generates a FlatIR subgraph for the operation and binds it to the specified
        inputs and outputs.

        Args:
            inputs: The inputs to the subgraph.
            outputs: The outputs of the subgraph.
        """
        ...

    def __str__(self) -> str:
        """
        Returns a Trace string representation of the operation.

        Returns:
            The Trace string representation of the operation.
        """
        assert len(self.outputs) == 1, "Base class implementation only works for single output operations!"

        args = [
            f"{field.name}={getattr(self, field.name)}"
            for field in dataclasses.fields(self)
            if field.name not in [base_field.name for base_field in dataclasses.fields(BaseOperator)]
        ]
        return f"{self.outputs[0].name} = {self.__class__.__name__.lower()}({', '.join([inp.name for inp in self.inputs] + args)})"
