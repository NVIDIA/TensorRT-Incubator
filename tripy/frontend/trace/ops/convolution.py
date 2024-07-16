from dataclasses import dataclass

from collections.abc import Sequence
from tripy.frontend.dim import dynamic_dim
from tripy.frontend.trace.ops.base import BaseTraceOp

import tripy.frontend.trace.ops.utils as op_utils


@dataclass(repr=False)
class Convolution(BaseTraceOp):
    padding: Sequence[Sequence[int]]
    stride: Sequence[int]
    groups: int
    lhs_dilation: Sequence[int]
    rhs_dilation: Sequence[int]

    def verify_spatial_rank(self, attr, rank, string):
        spatial_rank = rank - 2
        if attr and len(attr) != spatial_rank:
            op_utils.raise_error_io_info(
                self,
                f"Number of {string} values does not match number of spatial dimensions in the input.",
                details=[
                    f"Got {len(attr)} {string} value pairs but the number of spatial dimensions is: {spatial_rank}.",
                ],
            )

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def validate_inputs(self, tensor_shape, kernel_shape):
        if len(tensor_shape) != len(kernel_shape):
            op_utils.raise_error_io_info(
                self,
                "Input tensor and kernel must have the same rank.",
                details=[
                    f"Input tensor for operation: 'convolution' has shape: {tensor_shape} [rank = {len(tensor_shape)}], "
                    f"but should have the same rank as the kernel of shape: {kernel_shape} [rank = {len(kernel_shape)}]."
                ],
            )

        rank = len(tensor_shape)

        self.verify_spatial_rank(self.padding, rank, "padding")
        self.verify_spatial_rank(self.stride, rank, "stride")
        self.verify_spatial_rank(self.lhs_dilation, rank, "lhs_dilation")
        self.verify_spatial_rank(self.rhs_dilation, rank, "rhs_dilation")

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, "convolution")
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConvolutionOp

        ConvolutionOp.build(
            inputs,
            outputs,
            padding=self.padding,
            stride=self.stride,
            feature_group_count=self.groups,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.rhs_dilation,
        )
