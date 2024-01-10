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
    def to_trace_str(self, input_names: List[str], output_names: List[str]) -> str:
        """
        Returns a Trace string representation of the operation.

        Args:
            inputs_names: The names of the input tensor(s).
            output_names: The names of the output tensor(s).

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

    @abc.abstractmethod
    def infer_dtypes(self, input_dtypes: List[dtype]) -> List[dtype]:
        """
        Infers dtypes for the operation.

        Args:
            input_dtypes: The dtypes of the input tensor(s).

        Returns:
            The dtypes of the output tensor(s).
        """
        ...

    @abc.abstractmethod
    def to_flat_ir(self, flat_ir) -> None:
        """
        Generates FlatIR ops for the operation.

        Args:
            flat_ir: FlatIR parent graph where new ops are inserted.
        """
        ...

    def infer_devices(self, input_devices: List) -> List:
        """
        Infers output devices for the operation.

        Args:
            input_devices: The devices of the input tensor(s).

        Returns:
            The devices of the output tensor(s).
        """

        def _all_same(inputs: List):
            return all(inp == inputs[0] for inp in inputs)

        if len(input_devices) > 1:
            assert _all_same(input_devices), "Inputs are on different devices!"
        return [input_devices[0]]
