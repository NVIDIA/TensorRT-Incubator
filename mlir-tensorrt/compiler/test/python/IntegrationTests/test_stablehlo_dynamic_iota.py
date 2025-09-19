# RUN: %pick-one-gpu %PYTHON %s
# REQUIRES: tensorrt-version-ge-10.0
import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np
from mlir_tensorrt.compiler.dialects import builtin, func, stablehlo


def build_program(dtype, iota_dim):
    @builtin.module(sym_name=f"dynamic_iota_test_{dtype}")
    def mlir_module():
        DYNAMIC = ir.RankedTensorType.get_dynamic_size()
        i32 = ir.IntegerType.get_signless(32)
        shape_type = ir.RankedTensorType.get([2], i32)
        result_type = ir.RankedTensorType.get([DYNAMIC, 3], dtype)

        @func.func(shape_type)
        def main(shape):
            return stablehlo.dynamic_iota(result_type, shape, iota_dimension=iota_dim)

        main.func_op.arg_attrs = [
            ir.DictAttr.get(
                {
                    "tensorrt.value_bounds": ir.Attribute.parse(
                        "#tensorrt.shape_profile<min=[1, 3], opt= [64, 3], max = [128, 3]>"
                    )
                }
            )
        ]

    return mlir_module


def get_mlir_dtype(dtype):
    if dtype == np.int32:
        return ir.IntegerType.get_signless(32)
    elif dtype == np.int64:
        return ir.IntegerType.get_signless(64)
    elif dtype == np.float32:
        return ir.F32Type.get()
    else:
        raise Exception("unsupported dtype")


def build_exe(client, dtype, iota_dim):
    module = build_program(dtype=get_mlir_dtype(dtype), iota_dim=iota_dim)
    task = client.get_compilation_task(
        "stablehlo-to-executable",
        [
            "--tensorrt-builder-opt-level=0",
            "--tensorrt-strongly-typed=false",
        ],
    )
    task.run(module.operation)
    return compiler.translate_mlir_to_executable(module.operation)


def run_test(exe, dtype, iota_dim):
    client = runtime.RuntimeClient()
    devices = client.get_devices()
    if len(devices) == 0:
        return
    stream = devices[0].stream

    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)

    dynamic_size = 128

    arg0 = client.create_memref(
        np.asarray([dynamic_size, 3], dtype=np.int32),
        device=devices[0],
        stream=stream,
    )
    arg1 = client.create_memref(
        np.zeros(shape=(dynamic_size, 3), dtype=dtype),
        device=devices[0],
        stream=stream,
    )
    session.execute_function("main", in_args=[arg0], out_args=[arg1], stream=stream)
    data = np.asarray(client.copy_to_host(arg1, stream=stream))
    stream.sync()

    broadcast_shape = [dynamic_size, 3]
    iota_size = dynamic_size if iota_dim == 0 else 3
    iota_reshape = [1, 1]
    iota_reshape[iota_dim] = -1

    expected = np.linspace(0, iota_size - 1, num=iota_size, dtype=dtype).reshape(
        *iota_reshape
    ) * np.ones(broadcast_shape, dtype=dtype)
    np.testing.assert_array_equal(data, expected)


if __name__ == "__main__":
    with ir.Context() as context, ir.Location.unknown():
        client = compiler.CompilerClient(context)
        for dtype in [np.int64, np.int32, np.float32]:
            for iota_dim in [0, 1]:
                exe = build_exe(client, dtype, iota_dim)
                run_test(exe, dtype, iota_dim)
