from dataclasses import dataclass

from typing import Tuple
from tripy.frontend.dim import dynamic_dim
from tripy.frontend.trace.ops.base import BaseTraceOp

import tripy.frontend.trace.ops.utils as op_utils


@dataclass(repr=False)
class Convolution(BaseTraceOp):
    # TODO (#146): Add additional params like paddding, strides, grouping, dilation
    padding: Tuple[Tuple[int]]
    stride: Tuple[int]

    def infer_shapes(self):
        tensor_shape = self.inputs[0].shape
        kernel_shape = self.inputs[1].shape

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

        if len(self.padding) != rank - 2:
            op_utils.raise_error_io_info(
                self,
                "Number of padding values does not match number of spatial dimensions in the input.",
                details=[
                    f"Got {len(self.padding)} padding value pairs but the number of spatial dimensions is: {rank - 2}.",
                ],
            )

        if len(self.stride) != rank - 2:
            op_utils.raise_error_io_info(
                self,
                "Number of stride values does not match number of spatial dimensions in the input.",
                details=[
                    f"Got {len(self.stride)} stride values but the number of spatial dimensions is: {rank-2}.",
                ],
            )

        spatial_shape = ()
        for spatial_dim_tensor, spatial_dim_kernel, pad, stride in zip(
            tensor_shape[2:], kernel_shape[2:], self.padding, self.stride
        ):
            dim_val = (
                1 + (spatial_dim_tensor.runtime_value - spatial_dim_kernel.runtime_value + pad[0] + pad[1]) // stride
            )
            dim = dynamic_dim(dim_val)
            spatial_shape += (dim,)

        output_shape = (tensor_shape[0],) + (kernel_shape[0],) + spatial_shape

        self.outputs[0].shape = output_shape

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, "convolution")
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConvolutionOp

        ConvolutionOp.build(inputs, outputs, padding=self.padding, stride=self.stride)
