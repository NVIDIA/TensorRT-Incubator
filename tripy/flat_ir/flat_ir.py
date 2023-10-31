from typing import Dict, List, Sequence, Set
from collections import defaultdict

from tripy import frontend
from tripy.flat_ir.layer import FIRLayer
from tripy.flat_ir.tensor import FIRTensor
from tripy.types import ShapeInfo


class FlatIR:
    """
    A flattened representation of a computation expressed by one or more Tensors.
    """

    def __init__(self, tensors: Sequence[frontend.Tensor]) -> None:
        """
        Args:
            tensors: The tensor(s) to evaluate. These are effectively
                the desired outputs.

        Example:
        ::
            from tripy.frontend import Tensor
            from tripy.flat_ir import FlatIR

            a = Tensor([0])

            flat_ir = FlatIR([a])

            assert flat_ir.layers[0].inputs == []
            assert flat_ir.layers[0].outputs[0].name == "t0"
        """
        self.layers: List[FIRLayer] = []
        self.outputs: List[FIRTensor] = []
        # Dict to cache shape information of a Tensor
        self._shape_map: Dict[str, ShapeInfo] = {}

        _tensor_names: Dict[int, str] = defaultdict(lambda: None)

        def get_tensor_name(tensor):
            tid = id(tensor)
            if tid not in _tensor_names:
                _tensor_names[tid] = f"t{len(_tensor_names)}"
            return _tensor_names[tid]

        # Track exprs that are being traced to pretty print later
        incoming_exprs = list(tensors)

        exprs = list(tensors)
        seen_tensor_ids: Set[int] = set()
        # Store the producer of a tensor
        producer_dict: Dict[str, FIRLayer] = {}
        while exprs:
            head = exprs.pop(0)

            if id(head) in seen_tensor_ids:
                continue
            seen_tensor_ids.add(id(head))

            exprs.extend(head.inputs)
            self.layers.append(
                FIRLayer(
                    [FIRTensor(get_tensor_name(inp), inp._stack_info, ShapeInfo(), None) for inp in head.inputs],
                    [FIRTensor(get_tensor_name(head), head._stack_info, ShapeInfo(), None)],
                    head.op,
                )
            )

            for op in self.layers[-1].outputs:
                producer_dict[op.name] = self.layers[-1]

            if head in incoming_exprs:
                self.outputs.append(*self.layers[-1].outputs)

        # Use the producer cache to fill the information for all tensors.
        for l in self.layers:
            for inp in l.inputs:
                inp.producer = producer_dict[inp.name]

        # Reverse the order of the layers so they are topologically sorted
        self.layers = self.topological_sort()

        # Perform shape inference to fill shape information for all tensors.
        self.infer_shapes()

    def topological_sort(self) -> List[FIRLayer]:
        stack = list()
        visited_nodes = defaultdict(lambda: False)

        def add_to_stack(v, visited, stack):
            visited[id(v)] = True
            for ip in v.inputs:
                if id(ip.producer) not in visited:
                    add_to_stack(ip.producer, visited, stack)

            stack.append(v)

        for l in self.layers:
            if id(l) not in visited_nodes:
                add_to_stack(l, visited_nodes, stack)

        assert len(self.layers) == len(stack)
        return stack

    def __str__(self) -> str:
        layer_strs: List[str] = []
        for layer in self.layers:
            layer_strs.append(
                layer.op.to_flat_ir_str([inp.name for inp in layer.inputs], [out.name for out in layer.outputs])
            )
        layer_strs.append(f'outputs: {", ".join(out.name for out in self.outputs)}')
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
