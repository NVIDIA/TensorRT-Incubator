from typing import Any, Dict

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo

from tripy.backend.mlir.utils import collect_input_output, make_ir_context, value_param_to_ir_const
from tripy.flat_ir import FlatIR
from tripy.ops import BinaryElementwise, Value


def lower_flat_ir_to_mlir(flat_ir: FlatIR) -> ir.Module:
    """
    Lowers flatIR representation of a program into its equivalent StableHLO representation.
    Args:
        flat_ir: flatIR representation of a program.
    Returns:
        mlir Module which is functionally equivalent to the input flatIR program.
    """
    with make_ir_context() as ctx, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body) as ip:
            # Lets assume only one function with inline code (#9 will fix it)
            inputs, outputs = collect_input_output(flat_ir)
            inp_types = []
            out_types = [ir.RankedTensorType.get(outputs[0].shape, ir.F32Type.get())]
            ftype = ir.FunctionType.get(inp_types, out_types)
            # Todo: Function name should be a property of flatIR and used here.
            func_op = func_dialect.FuncOp("tripyFunc", ftype, ip=ip)
            entry_block = func_op.add_entry_block()
            with ir.InsertionPoint(entry_block):
                ops = []
                hlo_tensors: Dict[str, Any] = {}
                for l in flat_ir.layers:
                    if isinstance(l.op, Value):
                        hlo_tensors[l.outputs[0].name] = value_param_to_ir_const(l.op)
                    else:
                        assert isinstance(l.op, BinaryElementwise)
                        if l.op.kind == BinaryElementwise.Kind.SUM:
                            add_out = hlo.AddOp(*[hlo_tensors[ip.name] for ip in l.inputs])
                            hlo_tensors[l.outputs[0].name] = add_out
                            ops.append(add_out)
                        else:
                            assert False, "Only Operation.SUM is supported by MLIR backend."
                func_dialect.ReturnOp(ops[-1])
        return module
