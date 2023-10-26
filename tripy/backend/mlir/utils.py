import subprocess

from jax._src.lib.mlir import dialects, ir

from tripy.flat_ir import FlatIR
from tripy.logging import G_LOGGER
from tripy.ops import Storage
from tripy.util import log_time


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
    output, _ = result.communicate()
    if result.returncode != 0:
        G_LOGGER.error(f"Command failed with return code {result.returncode}")
    print(output)


def collect_input_output(flatIR: FlatIR):
    inputs = []
    for l in flatIR.layers:
        if len(l.inputs) == 0:
            if isinstance(l.op, Storage):
                inputs.append(l)

    # TODO (#7): This needs to return any outputs requested by the user in the FlatIR constructor.
    return inputs, flatIR.layers[-1].outputs
