from dataclasses import dataclass
from collections.abc import Sequence
from typing import Optional

from tripy import export, utils
from tripy.common import datatype
from tripy.frontend.module.module import Module
from tripy.frontend.module.parameter import Parameter

from tripy.common.exception import raise_error


@export.public_api(document_under="modules")
@dataclass
@utils.constant_fields(["dtype", "padding", "stride", "groups", "dilation"])
class Conv(Module):
    r"""
    Applies a convolution on the input tensor.

    With an input of shape :math:`(N, C_{\text{in}}, D_0,\ldots,D_n)` and
    output of shape :math:`(N, C_{\text{out}}, D_{0_{\text{out}}},\ldots,D_{n_{\text{out}}})`
    the output values are given by:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{Bias}_{C_{\text{out}}} +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the cross-correlation operator applied over the spatial
    dimensions of the input and kernel,
    :math:`N` is the batch dimension, :math:`C` is the channel dimension, and
    :math:`D_0,\ldots,D_n` are the spatial dimensions.
    """

    dtype: datatype.dtype
    r"""The data type to use for the convolution weights."""

    weight: Parameter
    r"""The kernel of shape :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}}, *\text{kernel_dims})`."""

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

    groups: int
    r"""
    The number of groups in a grouped convolution where the input and output channels are divided into ``groups`` groups.
    Each output group is connected only to its corresponding input group through the convolution kernel weights,
    and the outputs for each group are concatenated to produce the final result. This is in contrast to a standard convolution
    which has full connectivity between all input and output channels. Grouped convolutions reduce computational cost by
    a factor of ``groups`` and can benefit model parallelism and memory usage.
    Note that `in_channels` and `out_channels` must both be divisible by ``groups``.
    """

    dilation: Sequence[int]
    r"""
    A sequence of length :math:`M` indicating the number of zeros to insert between kernel weights across each spatial dimension,
    where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`. 
    This is known as the à trous algorithm and further downsamples the output by increasing the receptive field of the kernel.
    For each dimension with value :math:`x`, :math:`x-1` zeros are inserted between kernel weights. 
    """

    bias: Optional[Parameter]
    r"""
    The bias term to add to the output. The bias has a shape of :math:`(\text{Channels_{\text{out}}},)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_dims: Sequence[int],
        padding: Sequence[Sequence[int]] = None,
        stride: Sequence[int] = None,
        groups: int = None,
        dilation: Sequence[int] = None,
        bias: bool = True,
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
            groups: The number of groups in a grouped convolution where the input and output channels are divided into ``groups`` groups.
                Each output group is connected only to its corresponding input group through the convolution kernel weights,
                and the outputs for each group are concatenated to produce the final result. This is in contrast to a standard convolution
                which has full connectivity between all input and output channels. Grouped convolutions reduce computational cost by
                a factor of ``groups`` and can benefit model parallelism and memory usage.
                Note that `in_channels` and `out_channels` must both be divisible by ``groups``. Defaults to 1 (standard convolution).
            dilation: A sequence of length :math:`M` indicating the number of zeros to insert between kernel weights across each spatial dimension,
                where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
                This is known as the à trous algorithm and further downsamples the output by increasing the receptive field of the kernel.
                For each dimension with value :math:`x`, :math:`x-1` zeros are inserted between kernel weights.
            bias: Whether to add a bias term to the output or not. The bias has a shape of :math:`(\text{Channels_{\text{out}}},)`. Defaults to True.
            dtype: The data type to use for the convolution weights. Defaults to float32.

        .. code-block:: python
            :linenos:
            :caption: Basic Example

            input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
            conv = tp.Conv(in_channels=1, out_channels=1, kernel_dims=(2, 2), dtype=tp.float32)
            output = conv(input)

            conv_layer_torch = torch.nn.Conv2d(1, 1, 2) # doc: omit
            conv_layer_torch.weight.data = torch.from_numpy(conv.weight.numpy()) # doc: omit
            conv_layer_torch.bias.data = torch.ones(1) # doc: omit
            expected = conv_layer_torch(torch.from_numpy(input.numpy())) # doc: omit

            assert torch.allclose(torch.from_numpy(output.numpy()), expected)

        .. code-block:: python
            :linenos:
            :caption: Using Padding and Stride

            input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
            conv = tp.Conv(1, 1, (3, 3), padding=((1, 1), (1, 1)), stride=(3, 1), bias=False, dtype=tp.float32)
            output = conv(input)

            conv_layer_torch = torch.nn.Conv2d(1, 1, 2, padding=1, stride=(3, 1), bias=False) # doc: omit
            conv_layer_torch.weight.data = torch.from_numpy(conv.weight.numpy()) # doc: omit
            expected = conv_layer_torch(torch.from_numpy(input.numpy())) # doc: omit

            assert torch.allclose(torch.from_numpy(output.numpy()), expected)

        .. code-block:: python
            :linenos:
            :caption: Depthwise Convolution

            input = tp.reshape(tp.arange(18, dtype=tp.float32), (1, 2, 3, 3))
            conv = tp.Conv(2, 2, (3, 3), groups=2, bias=False, dtype=tp.float32)
            output = conv(input)

            conv_layer_torch = torch.nn.Conv2d(2, 2, 3, groups=2, bias=False) # doc: omit
            conv_layer_torch.weight.data = torch.from_numpy(conv.weight.numpy()) # doc: omit
            expected = conv_layer_torch(torch.from_numpy(input.numpy())) # doc: omit

            assert torch.allclose(torch.from_numpy(output.numpy()), expected)

        .. code-block:: python
            :linenos:
            :caption: Dilated Convolution (à trous algorithm)

            input = tp.reshape(tp.arange(9, dtype=tp.float32), (1, 1, 3, 3))
            conv = tp.Conv(1, 1, (2, 2), dilation=(2, 2), bias=False, dtype=tp.float32)
            output = conv(input)

            conv_layer_torch = torch.nn.Conv2d(1, 1, 2, dilation=2, bias=False) # doc: omit
            conv_layer_torch.weight.data = torch.from_numpy(conv.weight.numpy()) # doc: omit
            expected = conv_layer_torch(torch.from_numpy(input.numpy())) # doc: omit

            assert torch.allclose(torch.from_numpy(output.numpy()), expected)
        """

        super().__init__()
        from tripy.frontend.ops.tensor_initializers import arange, ones
        from tripy.frontend.trace.ops.reshape import reshape

        self.groups = utils.default(groups, 1)

        if self.groups <= 0:
            raise_error(
                "Feature group count must be a positive integer.",
                details=[f"Got feature group count: {self.groups}."],
            )

        if in_channels % self.groups or out_channels % self.groups:
            raise_error(
                "Feature group count must divide both input and output channel counts evenly.",
                details=[
                    f"Got feature group count: {self.groups} which is incompatible with input and output channel counts: {in_channels} and {out_channels}."
                ],
            )

        kernel_shape = (out_channels, in_channels // self.groups, *kernel_dims)
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

        self.dilation = utils.default(dilation, (1,) * (rank - 2))

        if not all(isinstance(d, int) and d > 0 for d in self.dilation):
            raise_error(
                "Non-positive dilation is not supported.",
                details=[f"Got dilation: {self.dilation} but all values must be integers greater than 0."],
            )

        if bias:
            self.bias = Parameter(ones((out_channels,), dtype=dtype))

        self.dtype = dtype

    def __call__(self, input: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            input: The input tensor.

        Returns:
            A tensor of the same data type as the input with a shape
            :math:`(N, \text{out_channels}, D_{0_{\text{out}}},\ldots,D_{n_{\text{out}}})`
            where :math:`D_{k_{\text{out}}} = \large \left\lfloor \frac{D_{k_{\text{in}}} + \text{padding}_{k_0} + \text{padding}_{k_1} - \text{dilation}_k \times (\text{kernel_dims}_k - 1) - 1}{\text{stride}_k} \right\rfloor + \normalsize 1`
        """
        from tripy.frontend.trace.ops.convolution import Convolution
        from tripy.frontend.trace.ops.reshape import reshape
        from tripy.frontend.trace.ops.convolution import Convolution

        x = Convolution.build(
            [input, self.weight],
            self.padding,
            self.stride,
            self.groups,
            None,  # lhs_dilation for transposed conv only
            self.dilation,
        )
        if hasattr(self, "bias"):
            # TODO: #174 - Get integers out of TP Array & Fix convolution bias reshaping for broadcasting
            out_channels = self.weight.shape[0].eval().byte_buffer.get()[0]
            rank = self.weight.rank
            bias_shape_to_broadcast = (1,) + (out_channels,) + (1,) * (rank - 2)
            self.bias = reshape(self.bias, bias_shape_to_broadcast)
            x += self.bias
        return x
