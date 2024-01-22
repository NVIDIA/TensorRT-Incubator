import copy
from typing import Any, Dict, List, Tuple

from mlir import ir
from mlir.dialects import func as func_dialect

from tripy import utils
from tripy.backend.mlir.utils import make_ir_context
from tripy.flat_ir.ops import BaseFIROp
from tripy.frontend.dim import Dim


class FlatIRTensorInfo:
    def __init__(self, shape: Tuple[Dim], dtype):
        self.shape = shape
        self.dtype = dtype


class FlatIRShapeInfo:
    def __init__(self, shape: Tuple[Dim]):
        self.shape = shape

    def is_a_subset_of(self, cached: "FlatIRShapeInfo"):
        return all(curr.is_a_subset_of(cached) for curr, cached in zip(self.shape, cached.shape))


class FlatIR:
    """
    A flattened low level representation of a computation graph which maps directly with StableHLO dialect.
    """

    def __init__(self):
        self.inputs: List["FIRTensor"] = []
        self.outputs: List["FIRTensor"] = []
        self.ops: List[BaseFIROp] = []

        self._tensor_map: Dict[str] = {}

    def __str__(self):
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

    def to_mlir(self):
        inputs_idx = {inp.name: idx for idx, inp in enumerate(self.inputs)}

        with make_ir_context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body) as ip:
                # Lets assume only one function with inline code (#9 will fix it)
                inp_types = [inp.to_mlir() for inp in self.inputs]
                out_types = [o.to_mlir() for o in self.outputs]
                ftype = ir.FunctionType.get(inp_types, out_types)
                # Todo: Function name should be a property of Trace and used here.
                func_op = func_dialect.FuncOp("main", ftype, ip=ip)
                entry_block = func_op.add_entry_block()
                with ir.InsertionPoint(entry_block):
                    ops = []
                    hlo_tensors: Dict[str, Any] = {}
                    # Initialize tensor dict with inputs
                    for inp in self.inputs:
                        hlo_tensors[inp.name] = entry_block.arguments[inputs_idx[inp.name]]
                    for l in self.ops:
                        operands = []
                        for inp in l.inputs:
                            operands.append(hlo_tensors[inp.name])
                        out_ops = l.to_mlir(operands)
                        ops.extend(out_ops)
                        hlo_tensors.update(zip([out.name for out in l.outputs], out_ops))

                    func_dialect.ReturnOp([hlo_tensors[o.name] for o in self.outputs])

                # Create tensorrt.shape_profile attribute for all function arguments
                arg_attrs: List[Dict[str, ir.Attribute]] = [{} for _ in range(len(entry_block.arguments))]

                for inp in self.inputs:
                    min_profile_list = inp.get_optimization_profile_list("min")
                    max_profile_list = inp.get_optimization_profile_list("max")
                    opt_profile_list = inp.get_optimization_profile_list("opt")

                    arg = {
                        "tensorrt.shape_profile": ir.Attribute.parse(
                            f"#tensorrt.shape_profile<min={min_profile_list}, opt={opt_profile_list}, max={max_profile_list}>"
                        )
                    }

                    if inp.name in inputs_idx:
                        arg_attrs[inputs_idx[inp.name]] = arg

                func_op.arg_attrs = ir.ArrayAttr.get([ir.DictAttr.get(attrs) for attrs in arg_attrs])

                # Append device location if outputs are on host
                res_attrs = []
                for output in self.outputs:
                    if output.device.kind == "cpu":
                        res_attrs.append(ir.Attribute.parse("{tensorrt.host_tensor}"))
                    else:
                        res_attrs.append(ir.DictAttr.get({}))
                func_op.res_attrs = ir.ArrayAttr.get(res_attrs)

            return module

    def register_tensor(self, tensor: "FIRTensor") -> "FIRTensor":
        """
        Registers a tensor with this FlatIR instance. If the tensor has no name, a name unique to this FlatIR will be assigned.

        Args:
            tensor: The tensor to register.
        """
        tensor.name = utils.default(tensor.name, f"t_inter{len(self._tensor_map)}")
        if tensor.name in self._tensor_map:
            return self._tensor_map[tensor.name]
        self._tensor_map[tensor.name] = tensor
        return tensor

    def integrate_subgraph(self, inputs: List["FIRTensor"], outputs: List["FIRTensor"]):
        """
        Integrates a subgraph delineated by the given inputs and outputs into this FlatIR.
        """
        stack = copy.copy(outputs)

        tensors = []
        ops = []

        while stack:
            tensor = stack.pop()
            op = tensor.producer
            stack.extend(inp for inp in op.inputs if inp not in inputs)
            tensors.append(tensor)
            ops.append(op)

        for tensor in tensors:
            self.register_tensor(tensor)
        # Need to append in reverse order to maintain topological sorting.
        self.ops.extend(reversed(ops))

    def io_shape_info(self):
        i_tensor_info = [FlatIRShapeInfo([s for s in i.shape]) for i in self.inputs]
        o_tensor_info = [FlatIRShapeInfo([s for s in o.shape]) for o in self.outputs]
        return (i_tensor_info, o_tensor_info)

    def io_tensor_info(self):
        i_tensor_info = [FlatIRTensorInfo([s for s in i.shape], i.dtype) for i in self.inputs]
        o_tensor_info = [FlatIRTensorInfo([s for s in o.shape], o.dtype) for o in self.outputs]
        return (i_tensor_info, o_tensor_info)
