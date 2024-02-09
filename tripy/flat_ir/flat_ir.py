import copy
from typing import Dict, List, Set

from tripy import utils
from tripy.flat_ir.ops import BaseFlatIROp


class FlatIR:
    """
    A flattened low level representation of a computation graph which maps directly with StableHLO dialect.
    """

    def __init__(self):
        self.inputs: List["FlatIRTensor"] = []
        self.outputs: List["FlatIRTensor"] = []
        self.ops: List[BaseFlatIROp] = []

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
        from mlir import ir
        from mlir.dialects import func as func_dialect

        from tripy.backend.mlir.utils import make_ir_context

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
                    hlo_ops: Dict[str, ir.Operation] = {}
                    # Initialize tensor dict with inputs
                    for index, inp in enumerate(self.inputs):
                        hlo_ops[inp.name] = entry_block.arguments[index]

                    for layer in self.ops:
                        input_ops = [hlo_ops[inp.name] for inp in layer.inputs]

                        layer_ops = layer.to_mlir(input_ops)

                        ops.extend(layer_ops)
                        hlo_ops.update(zip([out.name for out in layer.outputs], layer_ops))

                    func_dialect.ReturnOp([hlo_ops[o.name] for o in self.outputs])

                # Create tensorrt.shape_profile attribute for all function arguments
                arg_attrs: List[Dict[str, ir.Attribute]] = []

                for inp in self.inputs:
                    min_profile_list = inp.get_optimization_profile_list("min")
                    max_profile_list = inp.get_optimization_profile_list("max")
                    opt_profile_list = inp.get_optimization_profile_list("opt")

                    arg_attrs.append(
                        {
                            "tensorrt.shape_profile": ir.Attribute.parse(
                                f"#tensorrt.shape_profile<min={min_profile_list}, opt={opt_profile_list}, max={max_profile_list}>"
                            )
                        }
                    )

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

    def register_tensor(self, tensor: "FlatIRTensor") -> "FlatIRTensor":
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

    def integrate_subgraph(self, inputs: List["FlatIRTensor"], outputs: List["FlatIRTensor"]):
        """
        Integrates a subgraph delineated by the given inputs and outputs into this FlatIR.
        """
        seen_tensors: Set[int] = set()
        ops = []

        # Implements dfs search
        def register_tensor_and_collect_ops(tensor, seen_tensors):
            if id(tensor) not in seen_tensors:
                seen_tensors.add(id(tensor))
                self.register_tensor(tensor)

                op = tensor.producer

                for inp in op.inputs:
                    if inp not in inputs:
                        register_tensor_and_collect_ops(inp, seen_tensors)

                ops.append(op)

        for start_tensor in outputs:
            register_tensor_and_collect_ops(start_tensor, seen_tensors)

        # Rebind the ops to tensors from this FlatIR
        for op in ops:
            op.inputs = [self.register_tensor(inp) for inp in op.inputs]
            op.outputs = [self.register_tensor(out) for out in op.outputs]

        self.ops.extend(ops)
