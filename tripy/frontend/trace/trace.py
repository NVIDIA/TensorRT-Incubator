import copy
from typing import List, Sequence, Set

from tripy.frontend.trace.ops import BaseTraceOp
from tripy.frontend.trace.tensor import TraceTensor
from tripy.common.exception import raise_error
from tripy.common import logger


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

    def __init__(self, tensors: Sequence["tripy.Tensor"], inputs: Sequence["tripy.Tensor"] = []) -> None:
        """
        Args:
            tensors: The tensor(s) to evaluate. These are effectively the desired outputs.
            inputs: Input tensors in a jit function.
        """
        self.ops: List[BaseTraceOp] = []
        self.inputs: List[TraceTensor] = [inp.op.outputs[0] for inp in inputs]
        self.outputs: List[TraceTensor] = []

        exprs = [tensor.op for tensor in tensors]
        # Track outputs:
        output_ids = set(id(expr) for expr in exprs)
        input_op_ids = set(id(inp.op) for inp in inputs)
        seen_op_ids: Set[int] = set()

        # Check all tensors for duplicate names. We currently rely on tensor names being
        # unique in the trace/flatIR. We could potentially change this in the future to
        # automatically make names unique instead of complaining to the user, but it's better
        # for traceability if we use the names set by the user/frontend.
        _tensor_map = {}

        def check_name(tensor):
            if tensor.name in _tensor_map and (_tensor_map[tensor.name] is not tensor):
                raise_error(
                    f"Found distinct tensors with the same name: '{tensor.name}'.",
                    details=["Tensor: ", tensor, "has the same name as another tensor: ", _tensor_map[tensor.name]],
                )
            _tensor_map[tensor.name] = tensor

        while exprs:
            head = exprs.pop(0)

            if id(head) in seen_op_ids:
                continue
            seen_op_ids.add(id(head))

            for io in head.inputs + head.outputs:
                check_name(io)

            if id(head) not in input_op_ids:
                # not as an input
                self.ops.append(head)
                exprs.extend([inp.producer for inp in head.inputs])

            if id(head) in output_ids:
                self.outputs.extend(head.outputs)

        # Reverse the order of the layers so they are topologically sorted
        self.ops = self.topological_sort()

        # Perform shape/dtype/device inference to fill shape information for all tensors.
        self._infer_tensor_info()

        logger.trace(lambda: f"{self}\n")

    def topological_sort(self) -> List[BaseTraceOp]:
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
            inputs = [flat_ir.register_tensor(inp.to_flat_ir()) for inp in op.inputs]
            outputs = [flat_ir.register_tensor(out.to_flat_ir()) for out in op.outputs]
            # Pass shallow copies of inputs/outputs so that the op is free to modify them
            op.to_flat_ir(copy.copy(inputs), copy.copy(outputs))
            flat_ir.integrate_subgraph(inputs, outputs)

        logger.flat_ir(lambda: f"{flat_ir}\n")
        return flat_ir
