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

    def infer_shapes(self):
        tensor_shape = self.inputs[0].shape
        kernel_shape = self.inputs[1].shape
        self.validate_inputs(tensor_shape, kernel_shape)

        # For regular convolution, we account for the number of zeros inserted into the kernel with dilation.
        # The output size is then a function of the padding and kernel size, scaled by the stride.
        spatial_shape = ()
        for spatial_dim_tensor, spatial_dim_kernel, pad, window_stride, rhs_dilation in zip(
            tensor_shape[2:], kernel_shape[2:], self.padding, self.stride, self.rhs_dilation
        ):
            input_spatial_size = spatial_dim_tensor.runtime_value
            kernel_spatial_size = spatial_dim_kernel.runtime_value
            kernel_dilated = kernel_spatial_size + (rhs_dilation - 1) * (kernel_spatial_size - 1)
            dim_val = 1 + (input_spatial_size + pad[0] + pad[1] - kernel_dilated) // window_stride
            dim = dynamic_dim(dim_val)
            spatial_shape += (dim,)
            output_shape = (tensor_shape[0],) + (kernel_shape[0],) + spatial_shape

        self.outputs[0].shape = output_shape

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


@dataclass(repr=False)
class ConvolutionTranspose(Convolution):
    padding: Sequence[Sequence[int]]
    stride: Sequence[int]
    groups: int
    lhs_dilation: Sequence[int]
    rhs_dilation: Sequence[int]

    def infer_shapes(self):
        tensor_shape = self.inputs[0].shape
        kernel_shape = self.inputs[1].shape
        self.validate_inputs(tensor_shape, kernel_shape)

        if self.lhs_dilation and not all(s == 1 for s in self.stride):
            op_utils.raise_error_io_info(
                self,
                f"Window stride must be all ones for transposed convolution.",
                details=[
                    f"Got {self.stride} for window stride, but transposed convolution LHS dilation is: {self.lhs_dilation}.",
                ],
            )

        # For transpose convolution, we invert the calculation for the expected shape.
        spatial_shape = ()
        for i, (spatial_dim_tensor, spatial_dim_kernel, pad, lhs_dilation, rhs_dilation) in enumerate(
            zip(tensor_shape[2:], kernel_shape[2:], self.padding, self.lhs_dilation, self.rhs_dilation)
        ):
            input_spatial_size = spatial_dim_tensor.runtime_value
            kernel_spatial_size = spatial_dim_kernel.runtime_value
            # Padding has been modified at the module->trace level such that pad[i] = rhs_dilation * (kernel_spatial_size - 1) - original_pad[i]
            # Hence, pad[0] + pad[1] - rhs_dilation * (kernel_spatial_size - 1) = rhs_dilation * (kernel_spatial_size - 1) - original_pad[0] - original_pad[1]
            # See reference for shape inference calculation here: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
            dim_val = (
                (input_spatial_size - 1) * lhs_dilation + pad[0] + pad[1] - rhs_dilation * (kernel_spatial_size - 1) + 1
            )

            if dim_val < 1:
                op_utils.raise_error_io_info(
                    self,
                    f"Calculated output size for spatial dimension idx {i} is too small.",
                    details=[
                        f""""
                        Inferred shape of {dim_val} for spatial idx {i} with following input parameters:
                        input size: {input_spatial_size}, kernel_size: {kernel_spatial_size}, modified padding (see documentation): {pad}, lhs_dilation (stride): {lhs_dilation}, rhs_dilation: {rhs_dilation}.
                        """,
                    ],
                )

            dim = dynamic_dim(dim_val)
            spatial_shape += (dim,)

        out_channels = kernel_shape[0]
        output_shape = (tensor_shape[0],) + (out_channels,) + spatial_shape

        self.outputs[0].shape = output_shape
