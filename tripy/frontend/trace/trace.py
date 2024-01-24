import copy
from typing import List, Sequence, Set

from tripy.frontend.ops import BaseOperator
from tripy.frontend.tensor import Tensor
from tripy.frontend.trace.tensor import TraceTensor


class Trace:
    """
    A flattened representation of a computation graph expressed by one or more Tensors.
    """

    def __init__(self, tensors: Sequence[Tensor], inputs: Sequence[Tensor] = []) -> None:
        """
        Args:
            tensors: The tensor(s) to evaluate. These are effectively the desired outputs.
            inputs: Input tensors in a jit function.
        """
        self.layers: List[BaseOperator] = []
        self.inputs: List[TraceTensor] = [inp.op.outputs[0] for inp in inputs]
        self.outputs: List[TraceTensor] = []

        exprs = [tensor.op for tensor in tensors]
        # Track outputs:
        output_ids = set(id(expr) for expr in exprs)
        seen_op_ids: Set[int] = set()

        # Reset names each time we create a trace. This is a hack since we depend on
        # names being identical to identify structurally equivalent traces/flat_irs for JIT caching purposes.
        # TODO (#70): Remove this and instead use the tensor names set by the frontend.
        _tensor_names = {}

        def get_name(tensor):
            tensor_id = id(tensor)
            if tensor_id not in _tensor_names:
                _tensor_names[tensor_id] = f"t{len(_tensor_names)}"
            return _tensor_names[tensor_id]

        while exprs:
            head = exprs.pop(0)

            if id(head) in seen_op_ids:
                continue
            seen_op_ids.add(id(head))

            for io in head.inputs + head.outputs:
                io.name = get_name(io)

            if head.inputs or head.const_fold:
                # not as an input
                self.layers.append(head)
                exprs.extend([inp.producer for inp in head.inputs])

            if id(head) in output_ids:
                self.outputs.extend(head.outputs)

        # Reverse the order of the layers so they are topologically sorted
        self.layers = self.topological_sort()

        # Perform shape/dtype/device inference to fill shape information for all tensors.
        self.infer_tensor_info()

    def topological_sort(self) -> List[BaseOperator]:
        stack = list()
        visited_layer_ids = set()

        def add_to_stack(layer, stack):
            visited_layer_ids.add(id(layer))
            for ip in filter(lambda inp: inp not in self.inputs, layer.inputs):
                if ip.producer is not None and id(ip.producer) not in visited_layer_ids:
                    add_to_stack(ip.producer, stack)

            stack.append(layer)

        for layer in self.layers:
            if id(layer) not in visited_layer_ids:
                add_to_stack(layer, stack)

        assert len(self.layers) == len(stack)
        return stack

    def __str__(self) -> str:
        layer_strs: List[str] = []
        if len(self.inputs):
            layer_strs.append("inputs:")
        for inp in self.inputs:
            layer_strs.append(f"    {str(inp)}")
        for layer in self.layers:
            layer_strs.append(str(layer))
        layer_strs.append("outputs:")
        for out in self.outputs:
            layer_strs.append(f"    {str(out)}")
        return "\n".join(layer_strs)

    def __eq__(self, other: "Trace") -> bool:
        return self.layers == other.layers and self.inputs == other.inputs

    def infer_tensor_info(self):
        """
        Infers basic information, like shape, dtype, and device, for all tensors in the trace.
        """

        # Compute and cache shape information for all tensors
        for inp in self.inputs:
            inp.producer.infer_shapes()
            inp.producer.infer_dtypes()
            inp.producer.infer_devices()

        for layer in self.layers:
            layer.infer_shapes()
            layer.infer_dtypes()
            layer.infer_devices()

    def to_flat_ir(self):
        from tripy.flat_ir.flat_ir import FlatIR

        flat_ir = FlatIR()

        flat_ir.inputs = [flat_ir.register_tensor(inp.to_flat_ir()) for inp in self.inputs]
        flat_ir.outputs = [flat_ir.register_tensor(out.to_flat_ir()) for out in self.outputs]

        for layer in self.layers:
            inputs = [inp.to_flat_ir() for inp in layer.inputs]
            outputs = [out.to_flat_ir() for out in layer.outputs]
            # Pass shallow copies of inputs/outputs so that the layer is free to modify them
            layer.to_flat_ir(copy.copy(inputs), copy.copy(outputs))
            flat_ir.integrate_subgraph(inputs, outputs)

        return flat_ir
