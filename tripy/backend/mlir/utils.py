import time
import subprocess
import numpy as np

from jax._src.lib.mlir import dialects
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from tripy.frontend.flat_ir import FlatIR
from tripy.frontend.parameters.value import ValueParameters
from tripy.util.logging import G_LOGGER
from tripy.util.util import log_time


def make_ir_context() -> ir.Context:
    context = ir.Context()

    context.enable_multithreading(False)

    dialects.mhlo.register_mhlo_dialect(context)
    dialects.chlo.register_dialect(context)
    dialects.hlo.register_dialect(context)
    return context


@log_time
def execute_binary(bin_path):
    result = subprocess.Popen(
        bin_path, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, error = result.communicate()
    if result.returncode != 0:
        G_LOGGER.error(f"Command failed with return code {result.returncode}")
    print(output)


def value_param_to_ir_const(param):
    if type(param.values) == np.ndarray or type(param.values) == list:
        array = np.array(param.values, dtype=np.float32)
        attr = ir.DenseElementsAttr.get(np.ascontiguousarray(array), type=ir.F32Type.get(), shape=array.shape)
        return hlo.ConstantOp(attr).result


def collect_input_output(flatIR: FlatIR):
    inputs = []
    for l in flatIR.layers:
        if len(l.inputs) == 0:
            if type(l.params) == ValueParameters:
                inputs.append(l)

    return inputs, [flatIR.layers[-1].output]
