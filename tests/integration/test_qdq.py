import time

import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np

ASM = """
module {
  func.func @main() -> tensor<2x4xf32> {
    %0 = tensorrt.constant dense<1.000000e+00> : tensor<2x4xf32>
    %2 = tensorrt.constant dense<1.000000e+00> : tensor<1x4xf32>
    %3 = tensorrt.quantize in(%0 : tensor<2x4xf32>) scale(%2 : tensor<1x4xf32>) -> tensor<2x4xi4>
    %4 = tensorrt.dequantize in(%3 : tensor<2x4xi4>) scale(%2 : tensor<1x4xf32>) -> tensor<2x4xf32>
    return %4 : tensor<2x4xf32>
  }
}

"""


def test():
    # Build/parse the main function.
    with ir.Context() as context:
        m = ir.Module.parse(ASM)

        # Use the compiler API to compile to executable.
        client_opts = compiler.CompilerClientOptions()
        client = compiler.CompilerClient(context, client_opts)
        opts = compiler.StableHLOToExecutableOptions(tensorrt_builder_opt_level=3, tensorrt_strongly_typed=False)
        exe = compiler.compiler_stablehlo_to_executable(client, m.operation, opts)

    # The RuntimeClient can and should persist across multiple Executables, RuntimeSessions, etc.
    # It is primarily an interface for creating and manipulating buffers.
    client = runtime.RuntimeClient()
    stream = client.create_stream()
    devices = client.get_devices()

    if len(devices) == 0:
        return

    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)

    arg0 = client.create_memref(
        np.asarray([[-1, -2], [3, -2]], dtype=np.float32),
        dtype=runtime.ScalarTypeCode.f32,
        device=devices[0],
        stream=stream,
    )
    arg1 = client.create_memref(
        np.zeros(shape=(2, 2), dtype=np.float32),
        dtype=runtime.ScalarTypeCode.f32,
        device=devices[0],
        stream=stream,
    )
    session.execute_function("main", in_args=[], out_args=[arg1], stream=stream)

    data = np.asarray(client.copy_to_host(arg1, stream=stream))
    stream.sync()

    print(data)


if __name__ == "__main__":
    test()
