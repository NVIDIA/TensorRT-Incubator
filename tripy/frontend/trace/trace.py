import copy
from typing import List, Sequence, Set

from tripy.frontend.ops import BaseOperator
from tripy.frontend.tensor import Tensor
from tripy.frontend.trace.tensor import TraceTensor


class Trace:
    """
    A flattened representation of a computation graph expressed by one or more Tensors.
    """

    def _infer_tensor_info(self):
        """
        Infers basic information, like shape, dtype, and device, for all tensors in the trace.
        """

        # Compute and cache shape information for all tensors
        for inp in self.inputs:
            inp.producer.infer_shapes()
            inp.producer.infer_dtypes()
            inp.producer.infer_devices()

        for op in self.ops:
            op.infer_shapes()
            op.infer_dtypes()
            op.infer_devices()

    def __init__(self, tensors: Sequence[Tensor], inputs: Sequence[Tensor] = []) -> None:
        """
        Args:
            tensors: The tensor(s) to evaluate. These are effectively the desired outputs.
            inputs: Input tensors in a jit function.
        """
        self.ops: List[BaseOperator] = []
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
                self.ops.append(head)
                exprs.extend([inp.producer for inp in head.inputs])

            if id(head) in output_ids:
                self.outputs.extend(head.outputs)

        # Reverse the order of the layers so they are topologically sorted
        self.ops = self.topological_sort()

        # Perform shape/dtype/device inference to fill shape information for all tensors.
        self._infer_tensor_info()

    def topological_sort(self) -> List[BaseOperator]:
        stack = list()
        visited_layer_ids = set()

        def add_to_stack(op, stack):
            visited_layer_ids.add(id(op))
            for ip in filter(lambda inp: inp not in self.inputs, op.inputs):
                if ip.producer is not None and id(ip.producer) not in visited_layer_ids:
                    add_to_stack(ip.producer, stack)

            stack.append(op)

        for op in self.ops:
            if id(op) not in visited_layer_ids:
                add_to_stack(op, stack)

        assert len(self.ops) == len(stack)
        return stack

    def __str__(self) -> str:
        layer_strs: List[str] = []
        if len(self.inputs):
            layer_strs.append("inputs:")
        for inp in self.inputs:
            layer_strs.append(f"    {str(inp)}")
        for op in self.ops:
            layer_strs.append(str(op))
        layer_strs.append("outputs:")
        for out in self.outputs:
            layer_strs.append(f"    {str(out)}")
        return "\n".join(layer_strs)

    def __eq__(self, other: "Trace") -> bool:
        return self.ops == other.ops and self.inputs == other.inputs

    def to_flat_ir(self):
        from tripy.flat_ir.flat_ir import FlatIR

        flat_ir = FlatIR()

        flat_ir.inputs = [flat_ir.register_tensor(inp.to_flat_ir()) for inp in self.inputs]
        flat_ir.outputs = [flat_ir.register_tensor(out.to_flat_ir()) for out in self.outputs]

        for op in self.ops:
            inputs = [inp.to_flat_ir() for inp in op.inputs]
            outputs = [out.to_flat_ir() for out in op.outputs]
            # Pass shallow copies of inputs/outputs so that the op is free to modify them
            op.to_flat_ir(copy.copy(inputs), copy.copy(outputs))
            flat_ir.integrate_subgraph(inputs, outputs)

        return flat_ir
