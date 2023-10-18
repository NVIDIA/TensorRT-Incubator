from typing import List, Sequence, Dict, Set
import numpy as np

from tripy.frontend.flat_ir.layer import Layer
from tripy.frontend.flat_ir.tensor import Tensor
from tripy.frontend.tensor_expression import TensorExpression
from tripy.frontend.parameters.binary_elementwise import BinaryElementwiseParameters
from tripy.frontend.parameters.value import ValueParameters
from tripy.frontend.flat_ir.tensor import ShapeInfo


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
        # Dict to cache shape information of a Tensor
        self._shape_map: Dict[str, ShapeInfo] = {}

        _tensor_names: Dict[int, str] = {}

        def get_tensor_name(tensor):
            tid = id(tensor)
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
                f"Inputs: {[inp.__str__() for inp in layer.inputs]}\nOutput: '{layer.output.__str__()}'\nParameters: {layer.params}"
            )
        return "\n\n".join(layer_strs)

    def __eq__(self, other: "FlatIR") -> bool:
        return self.layers == other.layers

    def set_tensor_shape(self, tensor, shape):
        tensor.shape = shape
        if tensor.name not in shape:
            self._shape_map[tensor.name] = shape

    def infer_shapes(self):
        """
        Extremely naive shape inference routine.
        """
        # Compute and cache shape information for all tensors
        for layer in self.layers:
            if type(layer.params) == ValueParameters:
                self._shape_map[layer.output.name] = layer.params.shape()
            elif type(layer.params) == BinaryElementwiseParameters:
                assert (
                    len(layer.inputs) == 2
                ), f"Exepected BinaryElementwiseParameters to have 2 inputs, got {len(layer.inputs)}."
                ip1_shape = self._shape_map[layer.inputs[0].name]
                ip2_shape = self._shape_map[layer.inputs[1].name]
                assert (
                    ip1_shape == ip2_shape
                ), f"Input tensor shape for BinaryElementwiseParameters should be same. Got {ip1_shape}, {ip2_shape}"
                self._shape_map[layer.output.name] = ip1_shape

        # Assign cached shape information to corresponding Tensor
        for layer in self.layers:
            for ip in layer.inputs:
                ip.shape = self._shape_map[ip.name]

            layer.output.shape = self._shape_map[layer.output.name]
