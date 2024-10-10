# RUN: %PYTHON %s
import time

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
    session.execute_function(
        "main", in_args=[arg0], out_args=[arg1], stream=stream, client=client
    )

    data = np.asarray(client.copy_to_host(arg1, stream=stream))
    stream.sync()

    print(data)

    # Run execution a bunch more times asynchronously so that it calculates
    # `x * 2**num_iter`.
    num_iter = 5
    start_time = time.time()
    for _ in range(0, num_iter):
        session.execute_function(
            "main", in_args=[arg0], out_args=[arg0], stream=stream, client=client
        )
    data = np.asarray(client.copy_to_host(arg1, stream=stream))
    stream.sync()
    end_time = time.time()
    elapsed = end_time - start_time

    print(np.asarray(client.copy_to_host(arg0)))
    print(f"1000 iterations avg { (elapsed/num_iter)/1000.0} msec per iteration")


if __name__ == "__main__":
    stablehlo_add()

#      CHECK:   [ 0.  2.  4.  6.]
# CHECK-NEXT:   [ 8. 10. 12. 14.]
# CHECK-NEXT:   [16. 18. 20. 22.]]
# CHECK-NEXT:
# CHECK-NEXT:   [24. 26. 28. 30.]
# CHECK-NEXT:   [32. 34. 36. 38.]
# CHECK-NEXT:   [40. 42. 44. 46.]]]
# CHECK-NEXT:   [  0.  32.  64.  96.]
# CHECK-NEXT:   [128. 160. 192. 224.]
# CHECK-NEXT:   [256. 288. 320. 352.]]
# CHECK-NEXT:
# CHECK-NEXT:   [384. 416. 448. 480.]
# CHECK-NEXT:   [512. 544. 576. 608.]
# CHECK-NEXT:   [640. 672. 704. 736.]
