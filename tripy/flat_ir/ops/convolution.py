from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import stablehlo

from typing import Tuple
from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ConvolutionOp(BaseFlatIROp):
    def to_mlir(self, operands):
        # convolution spec: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution

        # Expected order of dimensions for operands is specified by dnums:
        # - Input tensor dimensions should be ordered as (n, f, d1,...,dN)
        # - Input kernel dimensions should be ordered as (o, i, d1,...,dN)
        # - Output dimensions will be ordered as (n, f, d1,...,dN)
        # - "n" is the batch dimension.
        # - "f" is the feature dimension.
        # - "i" and "o" are the input/output feature dimensions (respectively).
        # - "d1,...,dN" are spatial dimensions.
        lhs_shape = self.inputs[0].shape
        iota = tuple(range(len(lhs_shape)))
        lhs_spec, rhs_spec, out_spec = iota, iota, iota
        dnums = stablehlo.ConvDimensionNumbers.get(
            input_batch_dimension=lhs_spec[0],
            input_feature_dimension=lhs_spec[1],
            input_spatial_dimensions=list(lhs_spec[2:]),
            kernel_output_feature_dimension=rhs_spec[0],
            kernel_input_feature_dimension=rhs_spec[1],
            kernel_spatial_dimensions=list(rhs_spec[2:]),
            output_batch_dimension=out_spec[0],
            output_feature_dimension=out_spec[1],
            output_spatial_dimensions=list(out_spec[2:]),
        )

        out_type = self.outputs[0].to_mlir()

        output = stablehlo.convolution(
            result=out_type,
            lhs=operands[0],
            rhs=operands[1],
            dimension_numbers=dnums,
            feature_group_count=1,
            batch_group_count=1,
            window_strides=None,
            padding=None,
            lhs_dilation=None,
            rhs_dilation=None,
            window_reversal=None,
        )
        return [output]
