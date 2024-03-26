from dataclasses import dataclass
from typing import Tuple, Optional
from collections.abc import Sequence

from tripy import export, utils
from tripy.common import datatype
from tripy.frontend.module.module import Module
from tripy.frontend.module.parameter import Parameter

from tripy.common.exception import raise_error


@export.public_api(document_under="modules")
@dataclass
@utils.constant_fields(["dtype", "padding", "stride"])
class Conv(Module):
    r"""
    Applies a convolution on the input tensor.

    With an input of shape :math:`(N, C_{\text{in}}, D_0,\ldots,D_n)` and
    output of shape :math:`(N, C_{\text{out}}, D_{0_{\text{out}}},\ldots,D_{n_{\text{out}}})`
    the output values are given by:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) =
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the cross-correlation operator applied over the spatial
    dimensions of the input and kernel,
    :math:`N` is the batch dimension, :math:`C` is the channel dimension, and
    :math:`D_0,\ldots,D_n` are the spatial dimensions.
    """

    dtype: datatype.dtype
    r"""The data type to use for the convolution weights."""

    weight: Parameter
    r"""The kernel of shape :math:`[\text{out_channels}, \text{in_channels}, *\text{kernel_dims}]`."""

    padding: Sequence[Sequence[int]]
    r"""
    A sequence of pairs of integers of length :math:`M` indicating the zero padding
    to apply to the input along each spatial dimension before and after the dimension respectively,
    where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
    """

    stride: Sequence[int]
    r"""
    A sequence of length :math:`M` indicating the stride of convolution across each spatial dimension,
    where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_dims: Sequence[int],
        padding: Sequence[Sequence[int]] = None,
        stride: Sequence[int] = None,
        dtype: datatype.dtype = datatype.float32,
    ) -> None:
        r"""
        Args:
            in_channels: The number of channels in the input tensor.
            out_channels: The number of channels produced by the convolution.
            kernel_dims: The spatial shape of the kernel.
            padding: A sequence of pairs of integers of length :math:`M` indicating the zero padding
                to apply to the input along each spatial dimension before and after the dimension respectively,
                where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
                Defaults to all 0.
            stride: A sequence of length :math:`M` indicating the stride of convolution across each spatial dimension,
                where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
                Defaults to all 1.
            dtype: The data type to use for the convolution weights.

        .. code-block:: python
            :linenos:
            :caption: Basic Example

            input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
            conv = tp.Conv(in_channels=1, out_channels=1, kernel_dims=(2, 2), dtype=tp.float32)
            output = conv(input)

            conv_layer_torch = torch.nn.Conv2d(1, 1, 2, bias=False) # doc: omit
            conv_layer_torch.weight.data = torch.from_numpy(conv.weight.numpy()) # doc: omit
            expected = conv_layer_torch(torch.from_numpy(input.numpy())) # doc: omit

            assert torch.allclose(torch.from_numpy(output.numpy()), expected)

        .. code-block:: python
            :linenos:
            :caption: Using Padding and Stride

            input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
            conv = tp.Conv(1, 1, (3, 3), padding=((1, 1), (1, 1)), stride=(3, 1), dtype=tp.float32)
            output = conv(input)

            conv_layer_torch = torch.nn.Conv2d(1, 1, 2, padding=1, stride=(3, 1), bias=False) # doc: omit
            conv_layer_torch.weight.data = torch.from_numpy(conv.weight.numpy()) # doc: omit
            expected = conv_layer_torch(torch.from_numpy(input.numpy())) # doc: omit

            assert torch.allclose(torch.from_numpy(output.numpy()), expected)
        """
        # TODO (146): Add bias support in module

        super().__init__()
        from tripy.frontend.ops.tensor_initializers import arange
        from tripy.frontend.trace.ops.reshape import reshape

        kernel_shape = (out_channels, in_channels, *kernel_dims)
        self.weight = Parameter(reshape(arange(utils.volume(kernel_shape), dtype=dtype), kernel_shape))

        rank = len(kernel_shape)
        self.padding = utils.default(padding, tuple(((0, 0) for _ in range(rank - 2))))

        if not all(len(pad) == 2 for pad in self.padding):
            raise_error(
                f"Padding must be provided as a sequence of pairs of integers.",
                details=[f"Supplied padding attribute: {self.padding} contains sequences that are not of length 2."],
            )

        self.stride = utils.default(stride, (1,) * (rank - 2))

        if not all(s > 0 for s in self.stride):
            raise_error(
                "Non-positive stride is not supported.",
                details=[f"Got stride: {self.stride} but all values must be integers greater than 0."],
            )

        self.dtype = dtype

    def __call__(self, input: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            input: The input tensor.

        Returns:
            A tensor of the same data type as the input with a shape
            :math:`(N, \text{out_channels}, D_{0_{\text{out}}},\ldots,D_{n_{\text{out}}})`
            where :math:`D_{k_{\text{out}}} = \Large \left\lfloor \frac{D_{k_{\text{in}}} - \text{kernel_dims}_k + \text{padding}_{k_0} + \text{padding}_{k_1}}{\text{stride}_k} \right\rfloor + \normalsize 1`
        """
        from tripy.frontend import Tensor
        from tripy.frontend.trace.ops.convolution import Convolution

        return Tensor.build(
            [input, self.weight],
            Convolution,
            self.padding,
            self.stride,
        )
