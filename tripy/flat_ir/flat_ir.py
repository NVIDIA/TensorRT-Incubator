from typing import Dict, List, Sequence, Set

from tripy.flat_ir.layer import Layer
from tripy.flat_ir.tensor import Tensor
from tripy.frontend import TensorExpression
from tripy.types import ShapeInfo


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
            from tripy.frontend import TensorExpression
            from tripy.flat_ir import FlatIR

            a = TensorExpression.tensor([0])

            flat_ir = FlatIR([a])

            assert flat_ir.layers[0].inputs == []
            assert flat_ir.layers[0].outputs[0].name == "t0"
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
                    [Tensor(get_tensor_name(head), head._stack_info, ShapeInfo())],
                    head.op,
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
                layer.op.to_flat_ir_str([inp.name for inp in layer.inputs], [out.name for out in layer.outputs])
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
            out_shapes = layer.op.infer_shapes([self._shape_map[inp.name] for inp in layer.inputs])

            for out, shape in zip(layer.outputs, out_shapes):
                self._shape_map[out.name] = shape

        # Assign cached shape information to corresponding Tensor
        for layer in self.layers:
            for io in layer.inputs + layer.outputs:
                io.shape = self._shape_map[io.name]
