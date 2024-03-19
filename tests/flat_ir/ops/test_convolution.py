import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ConvolutionOp


@pytest.fixture
def flat_ir():
    input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
    input.name = "input"
    conv_layer = tp.Conv(3, 16, (5, 5), dtype=tp.float32)
    conv_layer.weight.name = "kernel"

    output = conv_layer(input)
    output.name = "output"

    trace = Trace([output])
    yield trace.to_flat_ir()


class TestConvolutionOp:
    def test_str(self, flat_ir):
        Conv = flat_ir.ops[-1]
        assert isinstance(Conv, ConvolutionOp)
        assert (
            str(Conv) == "output: [shape=(4, 16, 4, 4,), dtype=(float32), loc=(gpu:0)] = ConvolutionOp(input, kernel)"
        )

    def test_mlir(self, flat_ir):
        helper.check_mlir(
            flat_ir.to_mlir(),
            """
            module {
                func.func @main() -> tensor<4x16x4x4xf32> {
                    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4x3x8x8xf32>
                    %2 = stablehlo.iota dim = 0 : tensor<16x3x5x5xf32>
                    %3 = stablehlo.convolution(%1, %2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x3x8x8xf32>, tensor<16x3x5x5xf32>) -> tensor<4x16x4x4xf32>
                    return %3 : tensor<4x16x4x4xf32>
                }
            }
            """,
        )
