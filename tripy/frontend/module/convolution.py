from dataclasses import dataclass
from typing import Tuple

from tripy import utils
from tripy.common import datatype
from tripy.frontend.module.module import Module
from tripy.frontend.module.parameter import Parameter
from tripy.utils import export


@export.public_api(document_under="modules")
@dataclass
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

            input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
            conv_layer = tp.Conv(in_channels=1, out_channels=1, kernel_dims=(2, 2), dtype=tp.float32)
            output = conv_layer(input)

            conv_layer_torch = torch.nn.Conv2d(1, 1, 2, bias=False) # doc: omit
            conv_layer_torch.weight.data = torch.from_numpy(conv_layer.weight.numpy()) # doc: omit
            expected = conv_layer_torch(torch.from_numpy(input.numpy())) # doc: omit

            assert torch.allclose(torch.from_numpy(output.numpy()), expected)
        """
        super().__init__()
        from tripy.frontend.ops.tensor_initializers import arange
        from tripy.frontend.trace.ops.reshape import reshape

        self.dtype = dtype

        kernel_shape = (out_channels, in_channels, *kernel_dims)
        self.weight = Parameter(reshape(arange(utils.volume(kernel_shape), dtype=dtype), kernel_shape))

    def __call__(self, input: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            input: The input tensor.

        Returns:
            A tensor of the same data type as the input with a shape
            :math:`(N, \text{out_channels}, D_{0_{\text{out}}},\ldots,D_{n_{\text{out}}})`
            where :math:`D_{k_{\text{out}}} = D_{k_{\text{in}}} - \text{kernel_dims}[k] + 1`
        """
        from tripy.frontend import Tensor
        from tripy.frontend.trace.ops.convolution import Convolution

        return Tensor.build(
            [input, self.weight],
            Convolution,
        )
