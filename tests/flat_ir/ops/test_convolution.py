import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ConvolutionOp


@pytest.fixture(params=[None, ((2, 2), (2, 2))], ids=["no_padding", "padding"])
def padding(request):
    return request.param


@pytest.fixture(params=[None, (2, 2)], ids=["no_stride", "stride"])
def stride(request):
    return request.param


@pytest.fixture
def flat_ir(padding, stride):
    spatial_input = 8
    spatial_kernel = 5
    input = tp.ones((4, 3, spatial_input, spatial_input), dtype=tp.float32)
    input.name = "input"
    conv_layer = tp.Conv(3, 16, (spatial_kernel, spatial_kernel), padding=padding, stride=stride, dtype=tp.float32)
    conv_layer.weight.name = "kernel"

    output = conv_layer(input)
    output.name = "output"
    spatial_output = (spatial_input - spatial_kernel + 2 * padding[0][0]) // stride[0] + 1

    trace = Trace([output])
    yield (trace.to_flat_ir(), spatial_output)


@pytest.mark.parametrize("padding", [((0, 0), (0, 0)), ((2, 2), (2, 2))], indirect=True)
@pytest.mark.parametrize("stride", [(1, 1), (2, 2)], indirect=True)
class TestConvolutionOp:
    def test_str(self, flat_ir, padding, stride):
        Conv = flat_ir[0].ops[-1]
        assert isinstance(Conv, ConvolutionOp)
        spatial_shape = flat_ir[1]
        assert (
            str(Conv)
            == f"output: [shape=(4, 16, {spatial_shape}, {spatial_shape},), dtype=(float32), loc=(gpu:0)] = ConvolutionOp(input, kernel, padding={padding}, stride={stride})"
        )

    def test_mlir(self, flat_ir, padding, stride):
        spatial_shape = flat_ir[1]
        stride = list(stride)
        padding = [list(inner) for inner in padding]

        helper.check_mlir(
            flat_ir[0].to_mlir(),
            f"""
            module {{
                func.func @main() -> tensor<4x16x{spatial_shape}x{spatial_shape}xf32> {{
                    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4x3x8x8xf32>
                    %2 = stablehlo.iota dim = 0 : tensor<1200xf32>
                    %3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1200xf32>
                    %5 = stablehlo.multiply %2, %4 : tensor<1200xf32>
                    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<1200xf32>
                    %8 = stablehlo.add %5, %7 : tensor<1200xf32>
                    %9 = stablehlo.reshape %8 : (tensor<1200xf32>) -> tensor<16x3x5x5xf32>
                    %10 = stablehlo.convolution(%1, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {{stride = {stride}, pad = {padding}}} {{batch_group_count = 1 : i64, feature_group_count = 1 : i64}} : (tensor<4x3x8x8xf32>, tensor<16x3x5x5xf32>) -> tensor<4x16x{spatial_shape}x{spatial_shape}xf32>
                    return %10 : tensor<4x16x{spatial_shape}x{spatial_shape}xf32>
                }}
            }}
            """,
        )
