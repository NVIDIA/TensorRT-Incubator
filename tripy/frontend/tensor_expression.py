from typing import Any, List

from tripy import util
from tripy.ops import BaseOperator, BinaryElementwise, Value
from tripy.logging import G_LOGGER


class TensorExpression:
    """
    Represents an operation applied to zero or more input tensors.
    """

    # It is very important that this is the only entrypoint to creating a tensor expression.
    # We include logic here that needs to be applied to all tensor expressions.
    def __init__(self, inputs: "List[TensorExpression]", op: BaseOperator) -> None:
        """
        Args:
            inputs: The inputs to this expression.
            op: The operation being applied.
        """
        self.inputs = inputs
        self.op = op
        self._stack_info = util.get_stack_info()

    @staticmethod
    def tensor(values: Any) -> "TensorExpression":
        # TODO: This should accept a GPU-backed tensor
        return TensorExpression([], Value(values))

    def __add__(self, other) -> "TensorExpression":
        """
        Performs an elementwise sum with another tensor expression.

        Args:
            other: The tensor to add to this one.

        Returns:
            A tensor expression representing an elementwise sum.
        """
        return TensorExpression(
            [self, other],
            BinaryElementwise(BinaryElementwise.Kind.SUM),
        )

    def eval(self) -> None:
        from tripy.flat_ir import FlatIR
        from tripy.backend.mlir.__experimental_.compile import compile

        flatIR = FlatIR([self])
        G_LOGGER.ir_printer(f"flatIR :\n{flatIR}")
        compile(flatIR)

    def __repr__(self) -> str:
        return f"{self.eval()}"
