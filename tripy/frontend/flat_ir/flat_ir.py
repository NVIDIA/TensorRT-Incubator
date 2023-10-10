from typing import List, Sequence

from tripy.frontend.flat_ir.layer import Layer
from tripy.frontend.flat_ir.tensor import Tensor
from tripy.frontend.tensor_expression import TensorExpression


class FlatIR:
    """
    A flattened representation of a computation expressed by one or more TensorExpressions.
    """

    def __init__(self, tensor_expressions: Sequence[TensorExpression]):
        """
        Args:
            tensor_expressions: The tensor expressions to evaluate. These are effectively
                the desired outputs.

        Example:
        ::

            a = TensorExpression.tensor([0])

            flat_ir = FlatIR([a])

            assert flat_ir.layers[0].inputs == []
            assert flat_ir.layers[0].output.id == id(a)
        """
        self.layers: List[Layer] = []

        exprs = list(tensor_expressions)
        while exprs:
            head = exprs.pop(0)
            exprs.extend(head.inputs)
            self.layers.append(
                Layer(
                    [Tensor(id(inp), inp._stack_info) for inp in head.inputs],
                    Tensor(id(head), head._stack_info),
                    head.params,
                )
            )

        # Reverse the order of the layers so they are topologically sorted
        self.layers = list(reversed(self.layers))
