from typing import Any, Dict

from mlir import ir
from mlir.dialects import func as func_dialect

from tripy.backend.mlir.utils import make_ir_context
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
            inp_types = [inp.to_mlir() for inp in flat_ir.inputs]
            out_types = [o.to_mlir() for o in flat_ir.outputs]
            ftype = ir.FunctionType.get(inp_types, out_types)
            # Todo: Function name should be a property of flatIR and used here.
            func_op = func_dialect.FuncOp("main", ftype, ip=ip)
            entry_block = func_op.add_entry_block()
            with ir.InsertionPoint(entry_block):
                ops = []
                hlo_tensors: Dict[str, Any] = {}
                for l in flat_ir.layers:
                    operands = []
                    for inp in l.inputs:
                        if inp.name in flat_ir.inputs_idx:
                            operands.append(entry_block.arguments[flat_ir.inputs_idx[inp.name]])
                        else:
                            operands.append(hlo_tensors[inp.name])
                    out_ops = l.op.to_mlir(operands)

                    ops.extend(out_ops)
                    hlo_tensors.update(zip([out.name for out in l.outputs], out_ops))

                func_dialect.ReturnOp([hlo_tensors[o.name] for o in flat_ir.outputs])
        return module
