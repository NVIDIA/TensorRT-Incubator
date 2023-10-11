from typing import List, Sequence, Dict

from tripy.frontend.flat_ir.layer import Layer
from tripy.frontend.flat_ir.tensor import Tensor
from tripy.frontend.tensor_expression import TensorExpression


class FlatIR:
    """
    A flattened representation of a computation expressed by one or more TensorExpressions.
    """

    def __init__(self, tensor_expressions: Sequence[TensorExpression]) -> None:
        """
        Args:
            tensor_expressions: The tensor expressions to evaluate. These are effectively
                the desired outputs.

        Example:
        ::
            from tripy.frontend import TensorExpression, FlatIR

            a = TensorExpression.tensor([0])

            flat_ir = FlatIR([a])

            assert flat_ir.layers[0].inputs == []
            assert flat_ir.layers[0].output.name == "t0"
        """
        self.layers: List[Layer] = []

        _tensor_names: Dict[int, str] = {}

        def get_tensor_name(tensor):
            tid = id(tensor)
            if tid not in _tensor_names:
                _tensor_names[tid] = f"t{len(_tensor_names)}"
            return _tensor_names[tid]

        exprs = list(tensor_expressions)
        while exprs:
            head = exprs.pop(0)
            exprs.extend(head.inputs)
            self.layers.append(
                Layer(
                    [Tensor(get_tensor_name(inp), inp._stack_info) for inp in head.inputs],
                    Tensor(get_tensor_name(head), head._stack_info),
                    head.params,
                )
            )

        # Reverse the order of the layers so they are topologically sorted
        self.layers = list(reversed(self.layers))

    def __str__(self) -> str:
        layer_strs: List[str] = []
        for layer in self.layers:
            layer_strs.append(
                f"Inputs: {[inp.name for inp in layer.inputs]}\nOutput: '{layer.output.name}'\nParameters: {layer.params}"
            )
        return "\n\n".join(layer_strs)

    def __eq__(self, other: "FlatIR") -> bool:
        return self.layers == other.layers
