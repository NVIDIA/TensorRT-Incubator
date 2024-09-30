# RUN: %PYTHON %s
import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from ml_dtypes import float8_e4m3fn, bfloat16


@dataclass
class TestCase:
    ir: str
    in_args: List[runtime.MemRefValue]
    out_args: List[runtime.MemRefValue]
    reinterpret_type: Optional[type] = None


def test_stablehlo_add(
    tests: List[TestCase], runtime_client: runtime.RuntimeClient, stream: runtime.Stream
):
    for test in tests:
        # Build/parse the main function.
        with ir.Context() as context:
            m = ir.Module.parse(test.ir)

            # Use the compiler API to compile to executable.
            client = compiler.CompilerClient(context)
            opts = compiler.StableHLOToExecutableOptions(
                client,
                ["--tensorrt-builder-opt-level=3", "--tensorrt-strongly-typed=false"],
            )
            exe = compiler.compiler_stablehlo_to_executable(client, m.operation, opts)

        session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
        session = runtime.RuntimeSession(session_options, exe)

        session.execute_function(
            "main",
            in_args=test.in_args,
            out_args=test.out_args,
            stream=stream,
            client=runtime_client,
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
    stream = client.create_stream()
    devices = client.get_devices()
    if len(devices) == 0:
        print("No GPU device found!")
        exit()
    numpy_f8 = np.asarray([[0.4, 4.24], [6.61, 8.81]], dtype=float8_e4m3fn)
    numpy_f8_reinterpret_ui8 = numpy_f8.view(np.uint8, np.ndarray)
    numpy_bf16 = np.asarray([[0.4, 4.24], [6.61, 8.81]], dtype=bfloat16)
    numpy_bf16_reinterpret_ui16 = numpy_bf16.view(np.uint16, np.ndarray)
    numpy_bf16_zeros = np.zeros(shape=(2, 2), dtype=bfloat16)
    numpy_bf16_zeros_reinterpret_ui16 = numpy_bf16_zeros.view(np.uint16, np.ndarray)

    Tests = [
        TestCase(
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
        ),
        TestCase(
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
        ),
        TestCase(
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
        ),
        TestCase(
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


# FP8

#      CHECK:   [[ 0.8125,  8.    ]
# CHECK-NEXT:   [13.    , 18.    ]]
# CHECK-NEXT:   [[0.8125, 0.8125]
# CHECK-NEXT:   [0.8125, 0.8125]]
# CHECK-NEXT:   [[ 0.8125,  8.    ]
# CHECK-NEXT:   [13.    , 18.    ]]

# BF16

#      CHECK:   [[0.800781, 8.5]
# CHECK-NEXT:   [13.25, 17.625]]
# CHECK-NEXT:   [[0.800781, 0.800781]
# CHECK-NEXT:   [0.800781, 0.800781]]
# CHECK-NEXT:   [[0.800781, 8.5]
# CHECK-NEXT:   [[0.800781, 8.5]
# CHECK-NEXT:   [13.25, 17.625]]

# INT4

#      CHECK:   [[  5.  -6.]
# CHECK-NEXT:   [  3. -10.]]
# CHECK-NEXT:   [[ 4. -8.]
# CHECK-NEXT:   [ 4. -8.]]
