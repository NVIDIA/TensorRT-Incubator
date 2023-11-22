import abc
from typing import List
from tripy.types import ShapeInfo
from tripy.datatype import DataType


class BaseOperator(abc.ABC):
    @abc.abstractmethod
    def to_flat_ir_str(self, input_names: List[str], output_names: List[str]) -> str:
        """
        Returns a FlatIR string representation of the operation.

        Args:
            inputs_names: The names of the input tensor(s).
            output_names: The names of the output tensor(s).

        Returns:
            The FlatIR string representation of the operation.
        """
        ...

    @abc.abstractmethod
    def infer_shapes(self, input_shapes: List[ShapeInfo]) -> List[ShapeInfo]:
        """
        Infers shapes for the operation.

        Args:
            input_shapes: The shapes of the input tensor(s).

        Returns:
            The shapes of the output tensor(s).
        """
        ...

    @abc.abstractmethod
    def infer_dtypes(self, input_dtypes: List[DataType]) -> List[DataType]:
        """
        Infers dtypes for the operation.

        Args:
            input_dtypes: The dtypes of the input tensor(s).

        Returns:
            The dtypes of the output tensor(s).
        """
        ...

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
