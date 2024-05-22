from dataclasses import dataclass
from typing import Optional

from tripy import export, utils
from tripy.common import datatype
from tripy.frontend.module.module import Module
from tripy.frontend.module.parameter import Parameter, DefaultParameter


@export.public_api(document_under="modules")
@dataclass
@utils.constant_fields(["dtype", "quant_dtype"])
class Linear(Module):
    r"""
    Applies a linear transformation to the input:

    :math:`Linear(x) = xW^T + b`
    """

    dtype: datatype.dtype
    r"""The data type used to perform the operation"""

    weight: Parameter
    r"""The :math:`W` matrix of shape :math:`[\text{out_features}, \text{in_features}]`"""

    bias: Optional[Parameter]
    r"""The :math:`b` matrix of shape :math:`[1, \text{out_features}]`"""

    quant_dtype: Optional[datatype.dtype]
    r"""The quantization data type"""

    weight_scale: Optional[Parameter]
    r"""The quantization scale for weight"""

    input_scale: Optional[Parameter]
    r"""The quantization scale for input"""

    weight_quant_dim: Optional[int]
    r"""The quantization dimension for weight"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: datatype.dtype = datatype.float32,
        quant_dtype: datatype.dtype = None,
        weight_quant_dim: int = None,
    ) -> None:
        """
        Args:
            in_features: Size of input features.
            out_features: Size of output features.
            bias: Whether to include the bias term.
            dtype: The data type to use for the weight and bias parameters.
            quant_dtype: The data type for quantization.
            weight_quant_dim: Quantization dimension of weights.

        .. code-block:: python
            :linenos:
            :caption: Example

            linear = tp.Linear(3, 4)

            input = tp.iota((2, 3))
            output = linear(input)

            assert cp.from_dlpack(output).get().shape == (2, 4)

        .. code-block:: python
            :linenos:
            :caption: Quantized

            linear = tp.Linear(3, 4, quant_dtype=tp.int8, weight_quant_dim=0)
            linear.weight = tp.Parameter(tp.ones((4, 3)))
            linear.bias = tp.Parameter(tp.ones((4,)))
            weight_scale = [0.9] * 4
            linear.weight_scale = tp.Parameter(weight_scale)

            input = tp.iota((2, 3))
            output = linear(input)

            assert cp.from_dlpack(output).get().shape == (2, 4)
        """
        super().__init__()

        self.dtype = dtype

        # Replace with random weights when #74 is completed.
        self.weight = DefaultParameter((out_features, in_features), dtype=dtype)

        if bias:
            self.bias = DefaultParameter((out_features,), dtype=dtype)

        self.quant_dtype = quant_dtype
        self.weight_quant_dim = weight_quant_dim
        if quant_dtype is not None:
            self._tripy_params["weight_scale"] = None
            self._tripy_params["input_scale"] = None

    def __call__(self, x: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            x: The input tensor, of shape :math:`[*, \text{in_features}]`.

        Returns:
            A tensor of shape :math:`[*, \text{out_features}]`.
        """
        from tripy.common.exception import raise_error
        from tripy.frontend.trace.ops.permute import transpose
        from tripy.frontend.trace.ops.unsqueeze import unsqueeze
        from tripy.frontend.trace.ops.quantize import quantize
        from tripy.frontend.trace.ops.dequantize import dequantize

        if self.quant_dtype is not None:
            if self.input_scale:
                if self.weight_quant_dim == 1:
                    # TODO(#157): Give more informative error message to explain why
                    #             it is not supported.
                    raise_error(
                        "Unsupported quantization parameters for Linear module.",
                        [
                            "weight_quant_dim cannot be 1 when input_scale is provided. ",
                            f"input_scale={self.input_scale}, weight_quant_dim={self.weight_quant_dim}",
                        ],
                    )
                q_x = quantize(x, self.input_scale, self.quant_dtype)
                x = dequantize(q_x, self.input_scale, self.dtype)

            q_weight = quantize(self.weight, self.weight_scale, self.quant_dtype, self.weight_quant_dim)
            weight = dequantize(q_weight, self.weight_scale, self.dtype, self.weight_quant_dim)
        else:
            weight = self.weight

        out = x @ (transpose(weight, 1, 0))
        if hasattr(self, "bias"):
            out = out + unsqueeze(self.bias, 0)

        return out
