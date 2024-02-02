import abc
from dataclasses import dataclass
from typing import List, Set

from tripy import utils


@dataclass(repr=False)
class BaseOperator(abc.ABC):
    inputs: List["TraceTensor"]
    """The inputs of this layer"""

    outputs: List["TraceTensor"]
    """The outputs of this layer"""

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

    def str_skip_fields(self) -> Set[str]:
        """
        Returns names of dataclass fields to skip when generating a string representation of the op.
        """
        return set()

    def __str__(self) -> str:
        """
        Returns a Trace string representation of the operation.

        Returns:
            The Trace string representation of the operation.
        """
        assert len(self.outputs) == 1, "Base class implementation only works for single output operations!"

        skip_fields = self.str_skip_fields()
        args = [
            f"{field.name}={getattr(self, field.name)}"
            for field in utils.get_dataclass_fields(self, BaseOperator)
            if field.name not in skip_fields
        ]
        return f"{self.outputs[0].name} = {self.__class__.__name__.lower()}({', '.join([inp.name for inp in self.inputs] + args)})"

    def __repr__(self) -> str:
        # This is a hack to prevent printing the entire stack info when we print this.
        return str(self)
