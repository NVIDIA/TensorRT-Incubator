from tripy.frontend.parameters import BaseParameters, BinaryElementwiseParameters, ValueParameters
from typing import List, Any


class TensorExpression:
    """
    Represents an operation applied to zero or more input tensors.
    """

    def __init__(self, inputs: "List[TensorExpression]", params: BaseParameters) -> None:
        """
        Args:
            inputs: The inputs to this expression.
            params: The parameters that describe the operation being applied.
        """
        self.inputs = inputs
        self.params = params

    @staticmethod
    def tensor(values: Any) -> "TensorExpression":
        # TODO: This should accept a GPU-backed tensor
        return TensorExpression([], ValueParameters(values))

    def __add__(self, other) -> "TensorExpression":
        """
        Performs an elementwise sum with another tensor expression.

        Args:
            other: The tensor to add to this one.

        Returns:
            A tensor expression representing an elementwise sum.
        """
        return TensorExpression([self, other], BinaryElementwiseParameters(BinaryElementwiseParameters.Operation.SUM))
