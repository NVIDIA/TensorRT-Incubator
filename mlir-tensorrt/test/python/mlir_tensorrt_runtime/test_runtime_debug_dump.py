# RUN: %PYTHON %s 2>&1
# REQUIRES: host-has-at-least-1-gpus

import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np

ASM = """
func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %1 = stablehlo.add %arg0, %arg0 : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  func.return %1 : tensor<2x3x4xf32>
}
"""


def stablehlo_add():
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
        np.arange(0.0, 24.0, dtype=np.float32).reshape(2, 3, 4).data,
        device=devices[0],
        stream=stream,
    )
    arg1 = client.create_memref(
        np.zeros(shape=(2, 3, 4), dtype=np.float32).data,
        device=devices[0],
        stream=stream,
    )
    session.execute_function("main", in_args=[arg0], out_args=[arg1], stream=stream)

    data = np.asarray(client.copy_to_host(arg1, stream=stream))
    stream.sync()

    print(data)


def test_global_debug():
    print("Testing GlobalDebug functionality")

    # Test setting and getting the flag
    print("\nTesting flag property:")
    runtime.GlobalDebug.flag = True
    assert runtime.GlobalDebug.flag == True, "Flag should be True"
    print("Flag set to True: Passed")

    runtime.GlobalDebug.flag = False
    assert runtime.GlobalDebug.flag == False, "Flag should be False"
    print("Flag set to False: Passed")

    # Enable global debug flag and test various debug types.
    runtime.GlobalDebug.flag = True
    print("\nTesting set_types method:")

    # Test with a single type
    runtime.GlobalDebug.set_types("runtime")
    stablehlo_add()
    print("Set single debug type 'runtime': Passed")

    # Test with multiple types
    debug_types = ["allocator", "runtime"]
    runtime.GlobalDebug.set_types(debug_types)
    stablehlo_add()
    print("Set multiple debug types ['allocator', 'runtime']: Passed")

    print("\nAll tests passed successfully!")


test_global_debug()
