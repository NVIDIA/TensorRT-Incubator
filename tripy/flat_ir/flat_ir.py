from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Sequence, Set

from tripy import frontend
from tripy.common.datatype import DataType
from tripy.flat_ir.layer import FIRLayer
from tripy.flat_ir.tensor import FIRTensor
from tripy.common.types import ShapeInfo

FIRTensorInfo = namedtuple("FIRTensorInfo", ["shape", "dtype", "device"])


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
        self.inputs: List[FIRTensor] = []
        self.outputs: List[FIRTensor] = []
        # Dict to map input name to argument index
        self.inputs_idx: Dict[str, int] = {}
        # Dict to cache tensor information
        self._tensor_info_map: Dict[str, FIRTensorInfo] = {}

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

            as_input = (len(head.inputs) == 0) and (head.const_fold is False)
            if as_input:
                input_fir_tensor = FIRTensor(
                    get_tensor_name(head),
                    head._stack_info,
                    head.op.infer_shapes([])[0],
                    None,
                    head.op.infer_dtypes([])[0],
                    head.op.infer_devices([])[0],
                )
                self.inputs.append(input_fir_tensor)
                producer_dict[input_fir_tensor.name] = None
            else:
                exprs.extend(head.inputs)
                # Shapes/dtypes for intermediate tensors are inferred by infer_shapes_and_dtypes().
                self.layers.append(
                    FIRLayer(
                        [
                            FIRTensor(get_tensor_name(inp), inp._stack_info, ShapeInfo(), None, None, None)
                            for inp in head.inputs
                        ],
                        [FIRTensor(get_tensor_name(head), head._stack_info, ShapeInfo(), None, None, None)],
                        head.op,
                    )
                )

                for output in self.layers[-1].outputs:
                    producer_dict[output.name] = self.layers[-1]

            if head in incoming_exprs:
                if as_input:
                    self.outputs.append(self.inputs[-1])
                else:
                    self.outputs.append(*self.layers[-1].outputs)

        for idx, inp in enumerate(self.inputs):
            self.inputs_idx[inp.name] = idx

        # Use the producer cache to fill the information for all tensors.
        for l in self.layers:
            for inp in l.inputs:
                inp.producer = producer_dict[inp.name]

        # Reverse the order of the layers so they are topologically sorted
        self.layers = self.topological_sort()

        # Perform shape/dtype/device inference to fill shape information for all tensors.
        self.infer_tensor_info()

    def topological_sort(self) -> List[FIRLayer]:
        stack = list()
        visited_nodes = defaultdict(lambda: False)

        def add_to_stack(v, visited, stack):
            visited[id(v)] = True
            for ip in v.inputs:
                if ip.producer and id(ip.producer) not in visited:
                    add_to_stack(ip.producer, visited, stack)

            stack.append(v)

        for l in self.layers:
            if id(l) not in visited_nodes:
                add_to_stack(l, visited_nodes, stack)

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
                layer.op.to_flat_ir_str([inp.name for inp in layer.inputs], [out.name for out in layer.outputs])
            )
        layer_strs.append("outputs:")
        for out in self.outputs:
            layer_strs.append(f"    {str(out)}")
        return "\n".join(layer_strs)

    def __eq__(self, other: "FlatIR") -> bool:
        return self.layers == other.layers and self.inputs == other.inputs

    def infer_tensor_info(self):
        """
        Extremely naive shape inference routine.
        """
        # Compute and cache shape information for all tensors
        for inp in self.inputs:
            self._tensor_info_map[inp.name] = FIRTensorInfo(inp.shape, inp.dtype, inp.device)

        for layer in self.layers:
            out_shapes = layer.op.infer_shapes([self._tensor_info_map[inp.name].shape for inp in layer.inputs])
            out_dtypes = layer.op.infer_dtypes([self._tensor_info_map[inp.name].dtype for inp in layer.inputs])
            out_devices = layer.op.infer_devices([self._tensor_info_map[inp.name].device for inp in layer.inputs])

            for out, shape, dtype, device in zip(layer.outputs, out_shapes, out_dtypes, out_devices):
                self._tensor_info_map[out.name] = FIRTensorInfo(shape, dtype, device)

        # Assign cached shape information to corresponding Tensor
        for layer in self.layers:
            for io in layer.inputs + layer.outputs:
                io.shape = self._tensor_info_map[io.name].shape
                io.dtype = self._tensor_info_map[io.name].dtype
                io.device = self._tensor_info_map[io.name].device
