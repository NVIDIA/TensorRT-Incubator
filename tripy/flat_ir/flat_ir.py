from typing import Dict, List, Set

from tripy import utils


class FlatIR:
    """
    A flattened low level representation of a computation graph which maps directly with StableHLO dialect.
    """

    def __init__(self):
        self.inputs: List["FlatIRTensor"] = []
        self.outputs: List["FlatIRTensor"] = []
        self.ops: List["BaseFlatIROp"] = []

        self.tensor_map: Dict[str] = {}

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
        def to_mlir_impl():
            from mlir_tensorrt.compiler import ir
            from mlir_tensorrt.compiler.dialects import func as func_dialect

            from tripy.backend.mlir.utils import make_ir_context, make_tensor_location

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

                        for op in self.ops:
                            input_ops = [hlo_ops[inp.name] for inp in op.inputs]

                            with make_tensor_location(
                                [inp.name for inp in op.inputs],
                                [out.name for out in op.outputs],
                                op.trace_input_names,
                                op.trace_output_names,
                            ):
                                layer_ops = op.to_mlir(input_ops)

                            ops.extend(layer_ops)
                            hlo_ops.update(zip([out.name for out in op.outputs], layer_ops))

                        func_dialect.ReturnOp([hlo_ops[o.name] for o in self.outputs])

                    # Create tensorrt.shape_profile attribute for all function arguments
                    arg_attrs: List[Dict[str, ir.Attribute]] = []

                    # Returns a list filled with requested optimization profile information.
                    def get_optimization_profile_list(tensor, attr):
                        return [
                            getattr(s, attr) if s.is_dynamic_dim() else s.min for s in utils.make_list(tensor.shape)
                        ]

                    for inp in self.inputs:
                        min_profile_list = get_optimization_profile_list(inp, "min")
                        max_profile_list = get_optimization_profile_list(inp, "max")
                        opt_profile_list = get_optimization_profile_list(inp, "opt")

                        arg_attrs.append(
                            {
                                "tensorrt.shape_profile": ir.Attribute.parse(
                                    f"#tensorrt.shape_profile<min={min_profile_list}, opt={opt_profile_list}, max={max_profile_list}>"
                                )
                            }
                        )

                    func_op.arg_attrs = ir.ArrayAttr.get([ir.DictAttr.get(attrs) for attrs in arg_attrs])

                    # Append device location if outputs are on host as MLIR-TensorRT does not adhere to this constraint.
                    # (#155): Fix TensorKindAnalysis to ensure result tensors with attribute `tensorrt.host_tensor` are allocated on host.
                    res_attrs = []
                    for output in self.outputs:
                        if output.device.kind == "cpu":
                            res_attrs.append(ir.Attribute.parse("{tensorrt.host_tensor}"))
                        else:
                            res_attrs.append(ir.DictAttr.get({}))
                    func_op.res_attrs = ir.ArrayAttr.get(res_attrs)

                return module

        from tripy.backend.mlir.utils import redirect_stderr

        try:
            with redirect_stderr() as outfile:
                mlir = to_mlir_impl()
        except Exception as exc:
            from tripy.backend.mlir.compiler import map_error_to_user_code_and_raise

            outfile.flush()
            outfile.seek(0)
            stderr = outfile.read()

            map_error_to_user_code_and_raise(self, exc, stderr.decode())

        return mlir

    def register_tensor(self, tensor: "FlatIRTensor") -> "FlatIRTensor":
        """
        Registers a tensor with this FlatIR instance. If the tensor has no name, a name unique to this FlatIR will be assigned.

        Args:
            tensor: The tensor to register.
        """
        tensor.name = utils.default(tensor.name, f"t_inter{len(self.tensor_map)}")
        if tensor.name in self.tensor_map:
            return self.tensor_map[tensor.name]
        self.tensor_map[tensor.name] = tensor
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

                assert (
                    op is not None
                ), f"Tensor: {tensor} has no producer set. Did you use the constructor instead of the `build()` function?"

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
            op.trace_input_names = [inp.name for inp in inputs]
            op.trace_output_names = [out.name for out in outputs]

        self.ops.extend(ops)
