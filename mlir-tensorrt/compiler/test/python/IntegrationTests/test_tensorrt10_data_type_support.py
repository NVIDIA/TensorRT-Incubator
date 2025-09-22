# RUN: %pick-one-gpu %PYTHON %s | FileCheck %s
# REQUIRES: all-gpus-support-fp8
# REQUIRES: tensorrt-version-ge-10.0
from dataclasses import dataclass
from typing import List, Optional

import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np
from ml_dtypes import bfloat16, float8_e4m3fn


@dataclass
class TestCase:
    name: str
    ir: str
    in_args: List[runtime.MemRefValue]
    out_args: List[runtime.MemRefValue]
    reinterpret_type: Optional[type] = None


def test_stablehlo_add(
    tests: List[TestCase],
    runtime_client: runtime.RuntimeClient,
    stream: runtime.Stream,
):
    # Build/parse the main function.
    with ir.Context() as context:
        compiler_client = compiler.CompilerClient(context)
        for test in tests:
            print(test.name)
            m = ir.Module.parse(test.ir)
            task = compiler_client.get_compilation_task(
                "stablehlo-to-executable",
                ["--tensorrt-builder-opt-level=0", "--tensorrt-strongly-typed=false"],
            )
            task.run(m.operation)
            exe = compiler.translate_mlir_to_executable(m.operation)

            session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
            session = runtime.RuntimeSession(client, session_options, exe)

            session.execute_function(
                "main", in_args=test.in_args, out_args=test.out_args, stream=stream
            )
            output = [
                (
                    np.asarray(runtime_client.copy_to_host(e, stream=stream)).view(
                        test.reinterpret_type, np.ndarray
                    )
                    if test.reinterpret_type
                    else np.asarray(runtime_client.copy_to_host(e, stream=stream))
                )
                for e in test.out_args
            ]
            stream.sync()
            print(output)


if __name__ == "__main__":

    # The RuntimeClient can and should persist across multiple Executables, RuntimeSessions, etc.
    # It is primarily an interface for creating and manipulating buffers.
    client = runtime.RuntimeClient()
    devices = client.get_devices()
    if len(devices) == 0:
        print("No GPU device found!")
        exit()
    stream = devices[0].stream
    numpy_f8 = np.asarray([[0.4, 4.24], [6.61, 8.81]], dtype=float8_e4m3fn)
    numpy_f8_reinterpret_ui8 = numpy_f8.view(np.uint8, np.ndarray)
    numpy_bf16 = np.asarray([[0.4, 4.24], [6.61, 8.81]], dtype=bfloat16)
    numpy_bf16_reinterpret_ui16 = numpy_bf16.view(np.uint16, np.ndarray)
    numpy_bf16_zeros = np.zeros(shape=(2, 2), dtype=bfloat16)
    numpy_bf16_zeros_reinterpret_ui16 = numpy_bf16_zeros.view(np.uint16, np.ndarray)

    Tests = [
        TestCase(
            "fp8-1",
            """
            func.func @main() -> tensor<2x2xf32> {
                %arg0 = tensorrt.constant dense<[[0.4, 4.24],[6.61, 8.81]]> : tensor<2x2xf8E4M3FN>
                %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
                %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<2x2xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<2x2xf32>
                %1 = stablehlo.add %dq_arg0, %dq_arg0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                func.return %1: tensor<2x2xf32>
            }
            """,
            [],
            [
                client.create_memref(
                    np.zeros(shape=(2, 2), dtype=np.float32).data,
                    device=devices[0],
                    stream=stream,
                )
            ],
            # [np.array([[0.8125, 8.0], [13.0, 18.0]], dtype=np.float32)],
        ),
        TestCase(
            "fp8-2",
            """
            func.func @main() -> tensor<2x2xf32> {
                %arg0 = tensorrt.constant dense<0.4> : tensor<2x2xf8E4M3FN>
                %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
                %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<2x2xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<2x2xf32>
                %1 = stablehlo.add %dq_arg0, %dq_arg0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                func.return %1: tensor<2x2xf32>
            }
            """,
            [],
            [
                client.create_memref(
                    np.zeros(shape=(2, 2), dtype=np.float32).data,
                    device=devices[0],
                    stream=stream,
                )
            ],
            # [np.array([[0.8125, 0.8125], [0.8125, 0.8125]], dtype=np.float32)],
        ),
        TestCase(
            "fp8-3",
            """
            func.func @main(%arg0: tensor<2x2xf8E4M3FN>) -> tensor<2x2xf32> {
                %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
                %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<2x2xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<2x2xf32>
                %1 = stablehlo.add %dq_arg0, %dq_arg0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                func.return %1: tensor<2x2xf32>
            }
            """,
            [
                client.create_memref(
                    numpy_f8_reinterpret_ui8,
                    dtype=runtime.ScalarTypeCode.f8e4m3fn,
                    device=devices[0],
                    stream=stream,
                )
            ],
            [
                client.create_memref(
                    np.zeros(shape=(2, 2), dtype=np.float32).data,
                    device=devices[0],
                    stream=stream,
                )
            ],
            # [np.array([[0.8125, 8.0], [13.0, 18.0]], dtype=np.float32)],
        ),
        TestCase(
            "bf16-1",
            """
            func.func @main(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
                %1 = stablehlo.add %arg0, %arg0 : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
                func.return %1: tensor<2x2xbf16>
            }
            """,
            [
                client.create_memref(
                    numpy_bf16_reinterpret_ui16,
                    dtype=runtime.ScalarTypeCode.bf16,
                    device=devices[0],
                    stream=stream,
                )
            ],
            [
                client.create_memref(
                    numpy_bf16_zeros_reinterpret_ui16,
                    dtype=runtime.ScalarTypeCode.bf16,
                    device=devices[0],
                    stream=stream,
                )
            ],
            bfloat16,
        ),
        TestCase(
            "bf16-2",
            """
            func.func @main() -> tensor<2x2xbf16> {
                %arg0 = tensorrt.constant dense<0.4> : tensor<2x2xbf16>
                %1 = stablehlo.add %arg0, %arg0 : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
                func.return %1: tensor<2x2xbf16>
            }
            """,
            [],
            [
                client.create_memref(
                    numpy_bf16_zeros_reinterpret_ui16,
                    dtype=runtime.ScalarTypeCode.bf16,
                    device=devices[0],
                    stream=stream,
                )
            ],
            bfloat16,
        ),
        TestCase(
            "bf16-3",
            """
            func.func @main() -> tensor<2x2xbf16> {
                %arg0 = tensorrt.constant dense<[[0.4, 4.24],[6.61, 8.81]]> : tensor<2x2xbf16>
                %1 = stablehlo.add %arg0, %arg0 : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
                func.return %1: tensor<2x2xbf16>
            }
            """,
            [],
            [
                client.create_memref(
                    numpy_bf16_zeros_reinterpret_ui16,
                    dtype=runtime.ScalarTypeCode.bf16,
                    device=devices[0],
                    stream=stream,
                )
            ],
            bfloat16,
        ),
        TestCase(
            "i4-1",
            """
            func.func @main(%rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
                %scale = tensorrt.constant dense<1.0> : tensor<f32>
                %lhs = tensorrt.constant dense<[[1, 2],[3, 2]]> : tensor<2x2xi4>
                %dq_lhs = tensorrt.dequantize in (%lhs: tensor<2x2xi4>) scale (%scale: tensor<f32>) -> tensor<2x2xf32>
                %2 = tensorrt.matrix_multiply {
                op0 = #tensorrt.matrix_operation<kNONE>,
                op1 = #tensorrt.matrix_operation<kNONE>
                } ins(%dq_lhs, %rhs : tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                return %2 : tensor<2x2xf32>
            }
            """,
            [
                client.create_memref(
                    np.asarray([[-1, -2], [3, -2]], dtype=np.float32).data,
                    device=devices[0],
                    stream=stream,
                )
            ],
            [
                client.create_memref(
                    np.zeros(shape=(2, 2), dtype=np.float32).data,
                    device=devices[0],
                    stream=stream,
                )
            ],
        ),
        TestCase(
            "i4-2",
            """
            func.func @main(%rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
                %scale = tensorrt.constant dense<1.0> : tensor<f32>
                %lhs = tensorrt.constant dense<2> : tensor<2x2xi4>
                %dq_lhs = tensorrt.dequantize in (%lhs: tensor<2x2xi4>) scale (%scale: tensor<f32>) -> tensor<2x2xf32>
                %2 = tensorrt.matrix_multiply {
                op0 = #tensorrt.matrix_operation<kNONE>,
                op1 = #tensorrt.matrix_operation<kNONE>
                } ins(%dq_lhs, %rhs : tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                return %2 : tensor<2x2xf32>
            }
            """,
            [
                client.create_memref(
                    np.asarray([[-1, -2], [3, -2]], dtype=np.float32).data,
                    device=devices[0],
                    stream=stream,
                )
            ],
            [
                client.create_memref(
                    np.zeros(shape=(2, 2), dtype=np.float32).data,
                    device=devices[0],
                    stream=stream,
                )
            ],
        ),
    ]
    test_stablehlo_add(Tests, client, stream)


# CHECK: fp8-1
#      CHECK:   [ 0.8125,  8.    ]
# CHECK-NEXT:   [13.    , 18.    ]]

# CHECK: fp8-2
# CHECK-NEXT:   [0.8125, 0.8125]
# CHECK-NEXT:   [0.8125, 0.8125]]

# CHECK: fp8-3
# CHECK-NEXT:   [ 0.8125,  8.    ]
# CHECK-NEXT:   [13.    , 18.    ]]

# CHECK: bf16-1
# CHECK-NEXT:   [0.800781, 8.5]
# CHECK-NEXT:   [13.25, 17.625]]

# CHECK: bf16-2
# CHECK-NEXT:   [0.800781, 0.800781]
# CHECK-NEXT:   [0.800781, 0.800781]]

# CHECK: bf16-3
# CHECK-NEXT:   [0.800781, 8.5]
# CHECK-NEXT:   [13.25, 17.625]]

# CHECK: i4-1
# CHECK-NEXT:   [  5.,  -6.]
# CHECK-NEXT:   [  3., -10.]]

# CHECK: i4-2
# CHECK-NEXT:   [ 4., -8.]
# CHECK-NEXT:   [ 4., -8.]]
