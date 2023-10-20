import os
from typing import Any, Dict

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo
from jinja2 import Template

from tripy.backend.mlir.utils import collect_input_output, execute_binary, make_ir_context, value_param_to_ir_const
from tripy.flat_ir import FlatIR
from tripy.logging import G_LOGGER
from tripy.ops import BinaryElementwise, Value


def lower_flat_ir_to_mlir(flatIR: FlatIR):
    with make_ir_context() as ctx, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body) as ip:
            # Lets assume only one function with inline code
            inputs, outputs = collect_input_output(flatIR)
            ip_types = []
            op_types = [ir.RankedTensorType.get(outputs[0].shape, ir.F32Type.get())]
            ftype = ir.FunctionType.get(ip_types, op_types)
            func_op = func_dialect.FuncOp("tripyFunc", ftype, ip=ip)
            entry_block = func_op.add_entry_block()
            with ir.InsertionPoint(entry_block):
                ops = []
                hlo_tensors: Dict[str, Any] = {}
                for l in flatIR.layers:
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


def compile(flatIR: FlatIR):
    """Given a FlatIR, compile function traces the computation graph and generates an executable binary."""
    # Lower flatIR to corresponding StableHLO IR.
    mlir_module = lower_flat_ir_to_mlir(flatIR)
    textual = mlir_module.__str__()

    # Insert host code
    host_template = """
    func.func @main() -> index {
    %0 = func.call @tripyFunc() : () -> tensor<{{output_shape}}xf32>
    {% for i in range(repeat_count) %}
        %c_{{i}} = arith.constant {{i}} : index
        %1{{i}} = tensor.extract %0[%c_{{i}}] : tensor<{{output_shape}}xf32>
        executor.print "%f "(%1{{i}} : f32)
    {% endfor %}
    return %c_0 : index
    }
    """
    template = Template(host_template)
    output_shape = flatIR.layers[-1].outputs[0].shape[0]
    prefix = template.render(output_shape=output_shape, repeat_count=2)
    if textual.endswith("}\n"):
        textual = textual[:-2] + prefix + "}"

    G_LOGGER.ir_printer(f"StableHLO IR:\n{textual}")

    mlir_tensorrt = "mlir-tensorrt/"
    file_name = "tmp.mlir"
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "mlir-tensorrt/build/", file_name)

    with open(file_path, "w") as file:
        file.write(textual)

    cmd = """/tripy/mlir-tensorrt/build/tools/mlir-tensorrt-opt {} -pass-pipeline="builtin.module(func.func(convert-hlo-to-tensorrt{{allow-i64-to-i32-conversion}},tensorrt-expand-ops))" -mlir-elide-elementsattrs-if-larger=128 > /tripy/mlir-tensorrt/build/optimized_trt.mlir""".format(
        file_path
    )
    execute_binary(cmd)

    cmd = """/tripy/mlir-tensorrt/build/tools/mlir-tensorrt-opt -tensorrt-to-executor-pipeline -tensorrt-verbose=false -tensorrt-enable-timing-cache -disable-tensorrt-external-kernels -enable-tensorrt-builder-heuristics /tripy/mlir-tensorrt/build/optimized_trt.mlir > /tripy/mlir-tensorrt/build/translated.mlir"""
    execute_binary(cmd)

    cmd = """/tripy/mlir-tensorrt/build/tools/mlir-tensorrt-translate -mlir-to-runtime-executable /tripy/mlir-tensorrt/build/translated.mlir > /tripy/mlir-tensorrt/build/executable.mlir"""
    execute_binary(cmd)

    cmd = """/tripy/mlir-tensorrt/build/tools/mlir-tensorrt-runner -input-type=rtexe /tripy/mlir-tensorrt/build/executable.mlir"""
    execute_binary(cmd)

    if os.path.exists(file_path):
        os.remove(file_path)
