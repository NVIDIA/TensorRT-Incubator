from tripy.common import datatype
from tripy.frontend.module.module import Module
from tripy.frontend.module.parameter import Parameter
from tripy.utils import export
from typing import Tuple
from dataclasses import dataclass


@export.public_api(document_under="modules")
@dataclass
class Conv(Module):
    r"""
    Applies general convolution over a tensor with a kernel.

    The output value of the layer with input size
    :math:`(N, C_{\text{in}}, D_0,\ldots,D_n)` and output :math:`(N, C_{\text{out}}, D_{0_{\text{out}}},\ldots,D_{n_{\text{out}}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) =
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(B_i, k)


    where :math:`\star` is the valid cross-correlation operator for the spatial dimensions of the input and kernel,
    :math:`N` is a batch size, :math:`C` refers to the channel dimension, and
    :math:`D_0,\ldots,D_n` represent the spatial dimensions.

    Currently, additional options like padding, stride, dilation, and grouping
    are not supported.
    """

    dtype: datatype.dtype
    r"""The data type to use for the convolution weights."""

    weight: Parameter
    r"""The kernel of shape :math:`(\text{out_channels}, \text{in_channels}, *\text{kernel_dims})`."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_dims: Tuple[int], dtype: datatype.dtype = datatype.float32
    ) -> None:
        # TODO (146): Add bias support in module
        """
        Args:
            in_channels: The number of channels in the input tensor.
            out_channels: The number of channels produced by the convolution.
            kernel_dims: The spatial shape of the kernel.
            dtype: The data type to use for the convolution weights.

        .. code-block:: python
            :linenos:
            :caption: Example

            tensor = tp.reshape(tp.arange(12, dtype=tp.float32), (1, 3, 2, 2))
            conv_layer = tp.Conv(3, 4, (1, 1), dtype=tp.float32)
            output = conv_layer(tensor)

            conv_layer_torch = torch.nn.Conv2d(3, 4, 1, bias=False) # doc: omit
            conv_layer_torch.weight.data = torch.from_numpy(conv_layer.weight.numpy()) # doc: omit
            expected = conv_layer_torch(torch.from_numpy(tensor.numpy())) # doc: omit

            assert torch.allclose(torch.from_numpy(output.numpy()), expected)
        """
        super().__init__()
        from tripy.frontend.trace.ops.iota import iota

        self.dtype = dtype

        kernel_shape = (out_channels, in_channels, *kernel_dims)
        self.weight = Parameter(iota(kernel_shape, dim=0, dtype=dtype))

    def __call__(self, input: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            input: The input tensor.

        Returns:
            A tensor resulting from convolution of the input tensor with the kernel of the conv layer.
        """
        from tripy.frontend import Tensor
        from tripy.frontend.trace.ops.convolution import Convolution

        return Tensor.build(
            [input, self.weight],
            Convolution,
        )
