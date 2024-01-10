from collections import defaultdict, namedtuple
from typing import Dict, List, Sequence, Set

from tripy.frontend.ops import BaseOperator
from tripy.frontend.tensor import Tensor
from tripy.frontend.trace.tensor import TraceTensor

TraceTensorInfo = namedtuple("TraceTensorInfo", ["shape", "dtype", "device"])


class Trace:
    """
    A flattened representation of a computation graph expressed by one or more Tensors.
    """

    def __init__(self, tensors: Sequence[Tensor]) -> None:
        """
        Args:
            tensors: The tensor(s) to evaluate. These are effectively
                the desired outputs.
        """
        self.layers: List[BaseOperator] = []
        self.inputs: List[TraceTensor] = []
        self.outputs: List[TraceTensor] = []
        # Dict to cache tensor information
        self._tensor_info_map: Dict[str, TraceTensorInfo] = {}

        exprs = [tensor.op for tensor in tensors]
        # Track outputs:
        incoming_exprs = set(id(expr) for expr in exprs)
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

            if not head.inputs and not head.const_fold:
                # We stop tracing at input tensors.
                self.inputs.extend(head.outputs)
            else:
                self.layers.append(head)
                exprs.extend([inp.producer for inp in head.inputs])

            if id(head) in incoming_exprs:
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
            layer_strs.append(
                layer.to_trace_str([inp.name for inp in layer.inputs], [out.name for out in layer.outputs])
            )
        layer_strs.append("outputs:")
        for out in self.outputs:
            layer_strs.append(f"    {str(out)}")
        return "\n".join(layer_strs)

    def __eq__(self, other: "Trace") -> bool:
        return self.layers == other.layers and self.inputs == other.inputs

    def infer_tensor_info(self):
        """
        Extremely naive shape inference routine.
        """

        # Compute and cache shape information for all tensors
        for inp in self.inputs:
            inp.shape = inp.producer.infer_shapes([])[0]
            inp.dtype = inp.producer.infer_dtypes([])[0]
            inp.device = inp.producer.infer_devices([])[0]

            self._tensor_info_map[inp.name] = TraceTensorInfo(inp.shape, inp.dtype, inp.device)

        for layer in self.layers:
            out_shapes = layer.infer_shapes([self._tensor_info_map[inp.name].shape for inp in layer.inputs])
            out_dtypes = layer.infer_dtypes([self._tensor_info_map[inp.name].dtype for inp in layer.inputs])
            out_devices = layer.infer_devices([self._tensor_info_map[inp.name].device for inp in layer.inputs])

            for out, shape, dtype, device in zip(layer.outputs, out_shapes, out_dtypes, out_devices):
                self._tensor_info_map[out.name] = TraceTensorInfo(shape, dtype, device)

        # Assign cached shape information to corresponding Tensor
        for layer in self.layers:
            for io in layer.inputs + layer.outputs:
                io.shape = self._tensor_info_map[io.name].shape
                io.dtype = self._tensor_info_map[io.name].dtype
                io.device = self._tensor_info_map[io.name].device

    def to_flat_ir(self):
        from tripy.flat_ir.flat_ir import FlatIR

        flat_ir = FlatIR()

        flat_ir.inputs = [flat_ir.add_tensor(inp) for inp in self.inputs]
        flat_ir.outputs = [flat_ir.add_tensor(inp) for inp in self.outputs]

        for l in self.layers:
            l.to_flat_ir(flat_ir)

        return flat_ir
