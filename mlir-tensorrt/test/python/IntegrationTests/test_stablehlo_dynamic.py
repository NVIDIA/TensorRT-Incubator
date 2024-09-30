# RUN: %PYTHON %s | FileCheck %s
from typing import Iterable

import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np

program1 = """
module {
  func.func @main(%arg0: tensor<3x?x2xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [3, 2, 2], opt = [3, 4, 2], max = [3, 6, 2]>},
                  %arg1: tensor<3x?x2xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [3, 2, 2], opt = [3, 4, 2], max = [3, 6, 2]>}) -> tensor<3x?x2xf32> {
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<3x?x2xf32>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?x2xf32>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.get_dimension_size %arg0, dim = 2 : (tensor<3x?x2xf32>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.concatenate %1, %3, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %7 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<3x?x2xf32>) -> tensor<i32>
    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
    %9 = stablehlo.get_dimension_size %arg1, dim = 1 : (tensor<3x?x2xf32>) -> tensor<i32>
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.get_dimension_size %arg1, dim = 2 : (tensor<3x?x2xf32>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.concatenate %8, %10, %12, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %14 = stablehlo.maximum %6, %13 : tensor<3xi32>
    %15 = stablehlo.dynamic_broadcast_in_dim %arg0, %14, dims = [0, 1, 2] : (tensor<3x?x2xf32>, tensor<3xi32>) -> tensor<3x?x2xf32>
    %16 = stablehlo.dynamic_broadcast_in_dim %arg1, %14, dims = [0, 1, 2] : (tensor<3x?x2xf32>, tensor<3xi32>) -> tensor<3x?x2xf32>
    %17 = stablehlo.add %15, %16 : tensor<3x?x2xf32>
    return %17 : tensor<3x?x2xf32>
  }
}
"""

program2 = """
func.func @main(%arg0: tensor<?x2xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [2, 2], opt = [4, 2], max = [6, 2]>},
                %arg1: tensor<?x2xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [2, 2], opt = [4, 2], max = [6, 2]>})
                -> tensor<?x2xf32> {
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x2xf32>) -> tensor<i32>
  %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
  %2 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<?x2xf32>) -> tensor<i32>
  %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
  %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %5 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x2xf32>) -> tensor<i32>
  %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
  %7 = stablehlo.get_dimension_size %arg1, dim = 1 : (tensor<?x2xf32>) -> tensor<i32>
  %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
  %9 = stablehlo.concatenate %6, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %10 = stablehlo.maximum %4, %9 : tensor<2xi32>
  %11 = stablehlo.dynamic_broadcast_in_dim %arg0, %10, dims = [0, 1] : (tensor<?x2xf32>, tensor<2xi32>) -> tensor<?x2xf32>
  %12 = stablehlo.dynamic_broadcast_in_dim %arg1, %10, dims = [0, 1] : (tensor<?x2xf32>, tensor<2xi32>) -> tensor<?x2xf32>
  %13 = stablehlo.add %11, %12 : tensor<?x2xf32>
  return %13 : tensor<?x2xf32>
}
"""


def infer_output_shape(client, session, exe, input_shape):
    shape = (len(input_shape),)

    # Create place holder of output shape. Assume output rank is same as input rank.
    output_shape = [-1] * len(input_shape)

    # Allocate device memory to store input and output shape information. This is due to a limitation where all input and output shape tensors are device tensors.
    in_0 = np.array(input_shape, dtype=np.int64).view(np.uint8)
    in_1 = np.array(input_shape, dtype=np.int64).view(np.uint8)
    out_0 = np.array(output_shape, dtype=np.int64).view(np.uint8)

    # GPU device
    devices = client.get_devices()

    ins = [
        client.create_memref(in_0, shape=shape, dtype=runtime.ScalarTypeCode.i64),
        client.create_memref(in_1, shape=shape, dtype=runtime.ScalarTypeCode.i64),
    ]
    outs = [client.create_memref(out_0, shape=shape, dtype=runtime.ScalarTypeCode.i64)]

    session.execute_function(
        exe.get_signature("main").get_shape_func_name(),
        in_args=ins,
        out_args=outs,
        client=client,
    )

    # Copy output shape from device to host. Also, convert to int32 type since shape calculation uses int64 type.
    output_shape = np.asarray(outs[0])

    return output_shape


def test_program(program: str, input_shape: Iterable[int], debug: bool = True):
    # Build/parse the main function.
    with ir.Context() as context:
        m = ir.Module.parse(program)

        # Use the compiler API to compile to executable.
        client = compiler.CompilerClient(context)
        opts = compiler.StableHLOToExecutableOptions(
            client,
            [
                "--tensorrt-builder-opt-level=3",
                "--tensorrt-strongly-typed=false",
                "--entrypoint=main",
            ],
        )
        if debug:
            opts.set_debug_options(False, [], "tmp")
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
        np.ones(input_shape, dtype=np.float32).data,
        device=devices[0],
        stream=stream,
    )
    arg1 = client.create_memref(
        np.ones(input_shape, dtype=np.float32).data, device=devices[0], stream=stream
    )

    output_shape = infer_output_shape(client, session, exe, input_shape)

    arg2 = client.create_memref(
        np.zeros(output_shape, dtype=np.float32).data,
        device=devices[0],
        stream=stream,
    )

    session.execute_function(
        "main", in_args=[arg0, arg1], out_args=[arg2], stream=stream, client=client
    )
    data = np.asarray(client.copy_to_host(arg2, stream=stream))
    stream.sync()

    print(data)


if __name__ == "__main__":
    print("Test (3, ?, 2)")
    test_program(program1, (3, 4, 2))
    print("Test (?, 2)")
    test_program(program2, (4, 2))

# CHECK-LABEL: Test (3, ?, 2)
#       CHECK: [{{\[}}[2. 2.]
#       CHECK:   [2. 2.]
#       CHECK:   [2. 2.]
#       CHECK:   [2. 2.]]
#       CHECK:  {{\[}}[2. 2.]
#       CHECK:   [2. 2.]
#       CHECK:   [2. 2.]
#       CHECK:   [2. 2.]]
#       CHECK:  {{\[}}[2. 2.]
#       CHECK:   [2. 2.]
#       CHECK:   [2. 2.]
#       CHECK:   [2. 2.]]]

# CHECK-LABEL: Test (?, 2)
#       CHECK:  {{\[}}[2. 2.]
#       CHECK:  [2. 2.]
#       CHECK:  [2. 2.]
#       CHECK:  [2. 2.]]
