# RUN: %PYTHON %s
import time

import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np

single_return = """
func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %1 = stablehlo.add %arg0, %arg0 : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  func.return %1 : tensor<2x3x4xf32>
}
"""

scalar_return = """
func.func @main(%arg0: tensor<2x3x4xf32>) -> index {
  %1 = tensor.rank %arg0 : tensor<2x3x4xf32>
  func.return %1 : index
}
"""

mixed_return = """
func.func @main(%arg0: tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>, index) {
  %1 = stablehlo.add %arg0, %arg0 : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %2 = tensor.rank %1 : tensor<2x3x4xf32>
  func.return %1, %2 : tensor<2x3x4xf32>, index
}
"""

multiple_return = """
func.func @main(%arg0: tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>) {
  %1 = stablehlo.add %arg0, %arg0 : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %2 = stablehlo.add %arg0, %1 : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  func.return %1, %2 : tensor<2x3x4xf32>, tensor<2x3x4xf32>
}
"""

dynamic_shape = """
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

session_tracking_h2h = """
func.func @main() -> (tensor<?xi32, #plan.memory_space<host_pinned>> {tensorrt.host_tensor}) {
  %c = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
  %0 = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host_pinned>} : tensor<2xi32, #plan.memory_space<host_pinned>>
  %1 = bufferization.materialize_in_destination %c in %0 : (tensor<2xi32>, tensor<2xi32, #plan.memory_space<host_pinned>>) -> tensor<2xi32, #plan.memory_space<host_pinned>>
  %cast = tensor.cast %1 : tensor<2xi32, #plan.memory_space<host_pinned>> to tensor<?xi32, #plan.memory_space<host_pinned>>
  return %cast : tensor<?xi32, #plan.memory_space<host_pinned>>
}
"""

empty_shape_tensor = """
func.func @main() -> (tensor<?x?xi32, #plan.memory_space<host_pinned>> {tensorrt.host_tensor}) {
  %c = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %c_0 = stablehlo.constant dense<2> : tensor<i32>
  %c_1 = stablehlo.constant dense<1> : tensor<1xi32>
  %c_2 = stablehlo.constant dense<2> : tensor<1xi32>
  %c_3 = stablehlo.constant dense<2> : tensor<i32>
  %c_4 = stablehlo.constant dense<2> : tensor<1xi32>
  %0 = stablehlo.concatenate %c_2, %c_4, %c_1, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1 = stablehlo.dynamic_reshape %c, %0 : (tensor<2x2xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %c_5 = stablehlo.constant dense<2> : tensor<i32>
  %c_6 = stablehlo.constant dense<2> : tensor<1xi32>
  %c_7 = stablehlo.constant dense<2> : tensor<i32>
  %c_8 = stablehlo.constant dense<2> : tensor<1xi32>
  %c_9 = stablehlo.constant dense<0> : tensor<i32>
  %c_10 = stablehlo.constant dense<0> : tensor<1xi32>
  %2 = stablehlo.concatenate %c_6, %c_8, %c_10, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %3 = stablehlo.dynamic_broadcast_in_dim %1, %2, dims = [0, 1, 2] : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %c_11 = stablehlo.constant dense<2> : tensor<1xi32>
  %c_12 = stablehlo.constant dense<> : tensor<0xi32>
  %c_13 = stablehlo.constant dense<> : tensor<0xi32>
  %4 = stablehlo.compare  EQ, %c_12, %c_13 : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi1>
  %5 = stablehlo.select %4, %c_12, %c_12 : tensor<0xi1>, tensor<0xi32>
  %6 = stablehlo.dynamic_broadcast_in_dim %c_7, %5, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %7 = stablehlo.dynamic_broadcast_in_dim %c_9, %5, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %8 = stablehlo.multiply %6, %7 : tensor<i32>
  %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
  %10 = stablehlo.concatenate %c_11, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %11 = stablehlo.dynamic_reshape %3, %10 : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %11, %c0 : tensor<?x?xi32>
  %c1 = arith.constant 1 : index
  %dim_14 = tensor.dim %11, %c1 : tensor<?x?xi32>
  %12 = bufferization.alloc_tensor(%dim, %dim_14) {memory_space = #plan.memory_space<host_pinned>} : tensor<?x?xi32, #plan.memory_space<host_pinned>>
  %13 = bufferization.materialize_in_destination %11 in %12 : (tensor<?x?xi32>, tensor<?x?xi32, #plan.memory_space<host_pinned>>) -> tensor<?x?xi32, #plan.memory_space<host_pinned>>
  %cast = tensor.cast %13 : tensor<?x?xi32, #plan.memory_space<host_pinned>> to tensor<?x?xi32, #plan.memory_space<host_pinned>>
  return %cast : tensor<?x?xi32, #plan.memory_space<host_pinned>>
}
"""


# The RuntimeClient can and should persist across multiple Executables, RuntimeSessions, etc.
# It is primarily an interface for creating and manipulating buffers.
client = runtime.RuntimeClient()
stream = client.create_stream()
devices = client.get_devices()


def compile_executable(program, debug=False):
    # Build/parse the main function.
    with ir.Context() as context:
        m = ir.Module.parse(program)

        # Use the compiler API to compile to executable.
        client = compiler.CompilerClient(context)
        c_opts = [
            "--tensorrt-builder-opt-level=0",
            "--entrypoint=main",
            "--force-entrypoints-return-allocs",
        ]
        task = client.get_compilation_task("stablehlo-to-executable", c_opts)
        task.run(m.operation)
        exe = compiler.translate_mlir_to_executable(m.operation)
        return exe


def test_single_return():
    exe = compile_executable(single_return)
    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)
    arg0 = client.create_memref(
        np.arange(0.0, 24.0, dtype=np.float32).reshape(2, 3, 4).data,
        device=devices[0],
        stream=stream,
    )
    results = session.execute_function(
        "main", in_args=[arg0], stream=stream, client=client
    )

    output = np.asarray(client.copy_to_host(results[0], stream=stream))
    stream.sync()

    print(output)


def test_scalar_return():
    exe = compile_executable(scalar_return)
    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)
    arg0 = client.create_memref(
        np.arange(0.0, 24.0, dtype=np.float32).reshape(2, 3, 4).data,
        device=devices[0],
        stream=stream,
    )
    results = session.execute_function(
        "main", in_args=[arg0], stream=stream, client=client
    )

    print(results[0].data)


def test_mixed_return():
    exe = compile_executable(mixed_return)
    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)
    arg0 = client.create_memref(
        np.arange(0.0, 24.0, dtype=np.float32).reshape(2, 3, 4).data,
        device=devices[0],
        stream=stream,
    )
    results = session.execute_function(
        "main", in_args=[arg0], stream=stream, client=client
    )

    assert type(results[0]) == runtime.MemRefValue
    assert type(results[1]) == runtime.ScalarValue

    output = np.asarray(client.copy_to_host(results[0], stream=stream))
    stream.sync()

    print(output)
    print(results[1].data)


def test_multiple_return():
    exe = compile_executable(multiple_return)
    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)
    arg0 = client.create_memref(
        np.arange(0.0, 24.0, dtype=np.float32).reshape(2, 3, 4).data,
        device=devices[0],
        stream=stream,
    )
    results = session.execute_function(
        "main", in_args=[arg0], stream=stream, client=client
    )

    output_0 = np.asarray(client.copy_to_host(results[0], stream=stream))
    output_1 = np.asarray(client.copy_to_host(results[1], stream=stream))

    stream.sync()

    print(output_0)
    print(output_1)


def test_dynamic_shape():
    exe = compile_executable(dynamic_shape)
    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)
    arg0 = client.create_memref(
        np.arange(0.0, 8.0, dtype=np.float32).reshape((4, 2)).data,
        device=devices[0],
        stream=stream,
    )
    arg1 = client.create_memref(
        np.ones((4, 2), dtype=np.float32).data, device=devices[0], stream=stream
    )

    results = session.execute_function(
        "main", in_args=[arg0, arg1], stream=stream, client=client
    )

    output = np.asarray(client.copy_to_host(results[0], stream=stream))
    stream.sync()

    print(output)


def test_session_tracking_d2h():
    exe = compile_executable(session_tracking_h2h)
    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)
    results = session.execute_function("main", in_args=[], stream=stream, client=client)
    stream.sync()
    print(np.asarray(results[0]))


def test_empty_shape_tensor():
    exe = compile_executable(empty_shape_tensor)
    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)
    results = session.execute_function("main", in_args=[], stream=stream, client=client)
    stream.sync()
    print(np.asarray(results[0]))


if __name__ == "__main__":
    print("Test: single return")
    test_single_return()
    # CHECK-LABEL: Test: single return
    # CHECK: [[[ 0.  2.  4.  6.]
    # CHECK:   [ 8. 10. 12. 14.]
    # CHECK:   [16. 18. 20. 22.]]
    # CHECK:
    # CHECK:  [[24. 26. 28. 30.]
    # CHECK:   [32. 34. 36. 38.]
    # CHECK:   [40. 42. 44. 46.]]]

    # print("Test: multiple return")
    # test_multiple_return()
    # # CHECK-LABEL: Test: multiple return
    # # CHECK: [[[ 0.  2.  4.  6.]
    # # CHECK:   [ 8. 10. 12. 14.]
    # # CHECK:   [16. 18. 20. 22.]]
    # # CHECK:
    # # CHECK:  [[24. 26. 28. 30.]
    # # CHECK:   [32. 34. 36. 38.]
    # # CHECK:   [40. 42. 44. 46.]]]
    # # CHECK: [[[ 0.  3.  6.  9.]
    # # CHECK:   [12. 15. 18. 21.]
    # # CHECK:   [24. 27. 30. 33.]]
    # # CHECK:
    # # CHECK:  [[36. 39. 42. 45.]
    # # CHECK:   [48. 51. 54. 57.]
    # # CHECK:   [60. 63. 66. 69.]]]

    # print("Test: dynamic shape")
    # test_dynamic_shape()
    # # CHECK-LABEL: Test: dynamic shape
    # # CHECK: [[1. 2.]
    # # CHECK:  [3. 4.]
    # # CHECK:  [5. 6.]
    # # CHECK:  [7. 8.]]

    # print("Test: device to host copy")
    # test_session_tracking_d2h()
    # # CHECK-LABEL: Test: device to host copy
    # # CHECK: [1 2]

    # print("Test: empty shape tensor")
    # test_empty_shape_tensor()
    # # CHECK-LABEL: Test: empty shape tensor
    # # CHECK: []

    # print("Test: scalar return")
    # test_scalar_return()
    # # CHECK-LABEL: Test: scalar return
    # # CHECK: 3
    # print("Test: mixed return")

    # test_mixed_return()
    # # CHECK-LABEL: Test: mixed return
    # # CHECK: [[[ 0.  2.  4.  6.]
    # # CHECK:   [ 8. 10. 12. 14.]
    # # CHECK:   [16. 18. 20. 22.]]
    # # CHECK:
    # # CHECK:  [[24. 26. 28. 30.]
    # # CHECK:   [32. 34. 36. 38.]
    # # CHECK:   [40. 42. 44. 46.]]]
    # # CHECK: 3
