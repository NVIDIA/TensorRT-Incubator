# REQUIRES: tensorrt-version-ge-10.9
# REQUIRES: has-gpu-sm-gte-10.0
# RUN: %pick-one-gpu %PYTHON %s | FileCheck %s

import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np

ASM = """
func.func @main() -> tensor<2x32xf32>{
    %input = tensorrt.constant dense<[[0.0, 0.3, 0.6, 1.0, 1.3, 1.6, 1.9, 2.3,
        2.6, 2.9, 3.2, 3.5, 3.9, 4.2, 4.5, 4.8, 5.2,
        5.5, 5.8, 6.1, 6.5, 6.8, 7.1, 7.4, 7.7, 8.1,
        8.4, 8.7, 9., 9.4, 9.7, 10.0],
        [3.0, 3.3, 3.6, 4.0, 3.3, 3.6, 3.9,
        3.3, 2.6, 2.9, 3.2, 3.5, 3.9, 4.2, 4.5, 4.8,
        -5.2, -5.5, -5.8, -6.1, -5.5, -5.8, 5.1, 5.4,
        5.7, 5.1, 5.4, 5.7, 6.0, 6.4, 4.7, 6.0]]> : tensor<2x32xf32>
    %double_quant_scale = tensorrt.constant dense<1.0> : tensor<f32>
    %out_f4, %scale_f8 = tensorrt.dynamic_quantize {axis = 1 : i32} in(%input : tensor<2x32xf32>) double_quant_scale(%double_quant_scale : tensor<f32>) -> tensor<2x32xf4E2M1FN>, tensor<2x2xf8E4M3FN>
    %dequantize_scale = tensorrt.dequantize in(%scale_f8 : tensor<2x2xf8E4M3FN>) scale(%double_quant_scale : tensor<f32>) -> tensor<2x2xf32>
    %dequantize_data = tensorrt.dequantize in(%out_f4 : tensor<2x32xf4E2M1FN>) scale(%dequantize_scale : tensor<2x2xf32>) -> tensor<2x32xf32>
    return %dequantize_data : tensor<2x32xf32>
}
"""


def compile(client, op):
    task = client.get_compilation_task(
        "tensorrt-to-executable",
        [
            "--tensorrt-builder-opt-level=0",
            "--tensorrt-workspace-memory-pool-limit=1024kB",
            "--force-entrypoints-return-allocs",
        ],
    )
    task.run(op)
    return compiler.translate_mlir_to_executable(op)


def test_fp4_quantization():
    # Build/parse the main function.
    with ir.Context() as context:
        m = ir.Module.parse(ASM)

        # Use the compiler API to compile to executable.
        client = compiler.CompilerClient(context)
        exe = compile(client, m.operation)

    # The RuntimeClient can and should persist across multiple Executables, RuntimeSessions, etc.
    # It is primarily an interface for creating and manipulating buffers.
    client = runtime.RuntimeClient()
    devices = client.get_devices()
    if len(devices) == 0:
        return
    stream = devices[0].stream

    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)

    results = session.execute_function("main", in_args=[], stream=stream, client=client)

    data = np.asarray(client.copy_to_host(results[0], stream=stream))
    stream.sync()

    print(data)


if __name__ == "__main__":
    print("TEST: TensorRT FP4 support.")
    test_fp4_quantization()

#         CHECK-LABEL: TEST: TensorRT FP4 support.
#      CHECK{LITERAL}: [[ 0.       0.40625  0.40625  0.8125   1.21875  1.625    1.625    2.4375
# CHECK-NEXT{LITERAL}:   2.4375   3.25     3.25     3.25     3.25     4.875    4.875    4.875
# CHECK-NEXT{LITERAL}:   4.875    4.875    6.5      6.5      6.5      6.5      6.5      6.5
# CHECK-NEXT{LITERAL}:   6.5      6.5      9.75     9.75     9.75     9.75     9.75     9.75   ]
# CHECK-NEXT{LITERAL}: [ 3.25     3.25     3.25     3.25     3.25     3.25     3.25     3.25
# CHECK-NEXT{LITERAL}:   2.4375   3.25     3.25     3.25     3.25     4.875    4.875    4.875
# CHECK-NEXT{LITERAL}:  -4.5     -4.5     -6.75    -6.75    -4.5     -6.75     4.5      4.5
# CHECK-NEXT{LITERAL}:   6.75     4.5      4.5      6.75     6.75     6.75     4.5      6.75   ]]
