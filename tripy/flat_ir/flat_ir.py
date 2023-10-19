from typing import Callable, Dict, List, Sequence, Set

from tripy.flat_ir.layer import Layer
from tripy.flat_ir.tensor import ShapeInfo, Tensor
from tripy.frontend.parameters import BaseParameters
from tripy.frontend.tensor_expression import TensorExpression
from tripy.util import FunctionRegistry


class FlatIR:
    """
    A flattened representation of a computation expressed by one or more TensorExpressions.
    """

    str_from_params = FunctionRegistry(Callable[[BaseParameters, List[str], str], str])
    """Maps parameter types to functions that return string representations of the operation given the parameters, input names, and output name."""

    shape_inference = FunctionRegistry(Callable[[BaseParameters, List[ShapeInfo]], ShapeInfo])
    """Maps parameter types to functions that return the shape of the output of an operation given the parameters and input shapes"""

    def __init__(self, tensor_expressions: Sequence[TensorExpression]) -> None:
        """
        Args:
            tensor_expressions: The tensor expressions to evaluate. These are effectively
                the desired outputs.

        Example:
        ::
            from tripy.frontend import TensorExpression
            from tripy.flat_ir import FlatIR

            a = TensorExpression.tensor([0])

            flat_ir = FlatIR([a])

            assert flat_ir.layers[0].inputs == []
            assert flat_ir.layers[0].output.name == "t0"
        """
        self.layers: List[Layer] = []
        # Dict to cache shape information of a Tensor
        self._shape_map: Dict[str, ShapeInfo] = {}

        _tensor_names: Dict[int, str] = {}

        def get_tensor_name(tensor_expression):
            tid = id(tensor_expression)
            if tid not in _tensor_names:
                _tensor_names[tid] = f"t{len(_tensor_names)}"
            return _tensor_names[tid]

        exprs = list(tensor_expressions)
        seen_tensor_ids: Set[int] = set()
        while exprs:
            head = exprs.pop(0)

            if id(head) in seen_tensor_ids:
                continue
            seen_tensor_ids.add(id(head))

            exprs.extend(head.inputs)
            self.layers.append(
                Layer(
                    [Tensor(get_tensor_name(inp), inp._stack_info, ShapeInfo()) for inp in head.inputs],
                    Tensor(get_tensor_name(head), head._stack_info, ShapeInfo()),
                    head.params,
                )
            )

        # Reverse the order of the layers so they are topologically sorted
        self.layers = list(reversed(self.layers))
        # Perform shape inference to fill shape information for all tensors.
        self.infer_shapes()

    def __str__(self) -> str:
        layer_strs: List[str] = []
        for layer in self.layers:
            layer_strs.append(
                FlatIR.str_from_params[type(layer.params)](
                    layer.params, [inp.name for inp in layer.inputs], layer.output.name
                )
            )
        return "\n".join(layer_strs)

    def __eq__(self, other: "FlatIR") -> bool:
        return self.layers == other.layers

    def infer_shapes(self):
        """
        Extremely naive shape inference routine.
        """
        # Compute and cache shape information for all tensors
        for layer in self.layers:
            self._shape_map[layer.output.name] = FlatIR.shape_inference[type(layer.params)](
                layer.params, [self._shape_map[inp.name] for inp in layer.inputs]
            )

        # Assign cached shape information to corresponding Tensor
        for layer in self.layers:
            for ip in layer.inputs:
                ip.shape = self._shape_map[ip.name]

            layer.output.shape = self._shape_map[layer.output.name]
