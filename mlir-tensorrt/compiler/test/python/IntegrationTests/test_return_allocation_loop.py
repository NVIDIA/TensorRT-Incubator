# RUN: %PYTHON %s
# Creates a program that requries ~1GB of memory to run.
# We execute it in a loop, and in each execution the program needs to allocate a new output buffer
# of size ~1GB.
# This test verifies that the allocated outputs are being garbage collected correctly.
import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np
import cupy as cp
import gc


ASM = """
func.func @main(%arg0: tensor<256x1024x1024xf32>) -> tensor<256x1024x1024xf32> {
  %cst = tensorrt.constant dense<1.0> : tensor<1x1x1xf32>
  %1 = tensorrt.element_wise <kSUM>(%arg0, %cst : tensor<256x1024x1024xf32>, tensor<1x1x1xf32>) -> tensor<256x1024x1024xf32>
  func.return %1 : tensor<256x1024x1024xf32>
}
"""


def compile(client, op):
    task = client.get_compilation_task(
        "tensorrt-to-executable",
        [
            "--tensorrt-builder-opt-level=0",
            "--force-entrypoints-return-allocs",
        ],
    )
    task.run(op)
    return compiler.translate_mlir_to_executable(op)


def test_memref_create_in_loop():
    print("compiling")

    # Build/parse the main function.
    with ir.Context() as context:
        m = ir.Module.parse(ASM)

        # Use the compiler API to compile to executable.
        client = compiler.CompilerClient(context)
        exe = compile(client, m.operation)

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
        np.ones(shape=(256, 1024, 1024), dtype=np.float32).data,
        device=devices[0],
        stream=stream,
    )

    print("running")

    for i in range(0, 20):
        gc.collect()

        print(f"exec {i}/20 step 0")
        out = session.execute_function(
            "main", in_args=[arg0], stream=stream, client=client
        )
        out = cp.from_dlpack(out[0])

        print(f"exec {i}/20 step 1")

        out2 = session.execute_function(
            "main", in_args=[client.from_dlpack(out)], stream=stream, client=client
        )
        out = cp.from_dlpack(out2[0])

    stream.sync()
    print(out)


if __name__ == "__main__":
    test_memref_create_in_loop()
