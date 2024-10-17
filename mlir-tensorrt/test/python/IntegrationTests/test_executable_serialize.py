# RUN: %PYTHON %s 2>&1
import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np


ASM = """
func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
  %1 = stablehlo.add %arg0, %cst : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
}
"""


def test_serialize(ASM):
    # Build/parse the main function.
    with ir.Context() as context:
        m = ir.Module.parse(ASM)

        # Use the compiler API to compile to executable.
        client = compiler.CompilerClient(context)
        opts = compiler.StableHLOToExecutableOptions(
            client,
            ["--tensorrt-builder-opt-level=3", "--tensorrt-strongly-typed=false"],
        )
        exe = compiler.compiler_stablehlo_to_executable(client, m.operation, opts)

    client = runtime.RuntimeClient()
    stream = client.create_stream()
    devices = client.get_devices()

    if len(devices) == 0:
        return

    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session0 = runtime.RuntimeSession(session_options, exe)

    arg0 = client.create_memref(
        np.arange(4, dtype=np.float32).reshape(2, 2).data,
        device=devices[0],
        stream=stream,
    )
    arg1 = client.create_memref(
        np.zeros(shape=(2, 2), dtype=np.float32).data,
        device=devices[0],
        stream=stream,
    )
    session0.execute_function(
        "main", in_args=[arg0], out_args=[arg1], stream=stream, client=client
    )
    output0 = np.asarray(client.copy_to_host(arg1, stream=stream))
    stream.sync()

    # Serialize executable and reconstruct it from the result
    serialized_exe = exe.serialize()
    assert isinstance(serialized_exe, bytes)
    exe_reconstructed = compiler.Executable(serialized_exe)

    session1 = runtime.RuntimeSession(session_options, exe_reconstructed)
    session1.execute_function(
        "main", in_args=[arg0], out_args=[arg1], stream=stream, client=client
    )
    output1 = np.asarray(client.copy_to_host(arg1, stream=stream))
    stream.sync()

    assert np.array_equal(output0, output1)


test_serialize(ASM)
