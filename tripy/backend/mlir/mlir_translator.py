from typing import Any, Dict

from mlir import ir
from mlir.dialects import func as func_dialect

from tripy.backend.mlir.utils import collect_input_output, make_ir_context
from tripy.flat_ir import FlatIR


def lower_flat_ir_to_mlir(flat_ir: FlatIR) -> ir.Module:
    """
    Lowers flatIR representation of a program into its equivalent StableHLO representation.
    Args:
        flat_ir: flatIR representation of a program.
    Returns:
        mlir Module which is functionally equivalent to the input flatIR program.
    """
    with make_ir_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body) as ip:
            # Lets assume only one function with inline code (#9 will fix it)
            _, outputs = collect_input_output(flat_ir)
            inp_types = []
            out_types = [ir.RankedTensorType.get(o.shape, ir.F32Type.get()) for o in outputs]
            ftype = ir.FunctionType.get(inp_types, out_types)
            # Todo: Function name should be a property of flatIR and used here.
            func_op = func_dialect.FuncOp("main", ftype, ip=ip)
            entry_block = func_op.add_entry_block()
            with ir.InsertionPoint(entry_block):
                ops = []
                hlo_tensors: Dict[str, Any] = {}
                for l in flat_ir.layers:
                    out_ops = l.op.to_mlir([hlo_tensors[inp.name] for inp in l.inputs])
                    ops.extend(out_ops)
                    hlo_tensors.update(zip([out.name for out in l.outputs], out_ops))

                func_dialect.ReturnOp([hlo_tensors[o.name] for o in flat_ir.outputs])
        return module
