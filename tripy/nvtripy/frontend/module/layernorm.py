#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass
from typing import Sequence, Union

from nvtripy import export, utils
from nvtripy.common import datatype
from nvtripy.common.exception import raise_error
from nvtripy.frontend.module.module import Module
from nvtripy.frontend.module.parameter import DefaultParameter
from nvtripy.frontend.tensor import Tensor

from nvtripy.frontend.ops import utils as op_utils
from nvtripy.utils import wrappers
from nvtripy.trace.ops.layernorm import LayerNorm as LayerNormOp


@wrappers.interface(
    dtype_constraints={"input": "T1", "weight": "T1", "bias": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def layernorm(
    input: "nvtripy.Tensor",
    weight: "nvtripy.Tensor",
    bias: "nvtripy.Tensor",
    eps: float,
) -> "nvtripy.Tensor":

    normalized_shape = weight.shape
    D = len(normalized_shape)
    input_rank = input.rank

    if input_rank < 2:
        raise_error(
            f"Input must have a rank of at least 2, but got input of rank: {input.rank}",
            details=[
                "Input is expected to have shape (N, *) where N is the batch size, and * represents any number of channel dimension + spatial dimensions"
            ],
        )

    from nvtripy.frontend.ops.reshape import reshape

    if input_rank > D:
        broadcast_shape = (1,) * (input_rank - D) + normalized_shape
        weight = reshape(weight, broadcast_shape)
        bias = reshape(bias, broadcast_shape)

    return op_utils.create_op(
        LayerNormOp,
        [input, weight, bias],
        normalized_shape=normalized_shape,
        eps=eps,
    )


@export.public_api(document_under="operations/modules")
@dataclass
@utils.wrappers.constant_fields(["dtype", "normalized_shape"])
class LayerNorm(Module):
    r"""
    Applies layer normalization over the input tensor:

    :math:`\text{LayerNorm}(x) = \Large \frac{x - \bar{x}}{ \sqrt{\sigma^2 + \epsilon}} \normalsize * \gamma + \beta`

    where :math:`\bar{x}` is the mean and :math:`\sigma^2` is the variance.

    The mean and standard deviation are calculated over the last :math:`D`
    dimensions, where :math:`D` is the dimension of :math:`\text{normalized_shape}`.
    """

    dtype: datatype.dtype
    r"""The data type used to perform the operation."""

    normalized_shape: Sequence[int]
    r"""Defines the shape of the input tensor that is to be normalized over."""

    weight: Tensor
    r"""The :math:`\gamma` parameter of shape :math:`\text{normalized_shape}`."""

    bias: Tensor
    r"""The :math:`\beta` parameter of shape :math:`\text{normalized_shape}`."""

    eps: float
    """A value added to the denominator to prevent division by zero."""

    def __init__(
        self, normalized_shape: Union[int, Sequence[int]], dtype: datatype.dtype = datatype.float32, eps: float = 1e-5
    ) -> None:
        r"""
        Args:
            normalized_shape: The size of the feature dimension of the input over which normalization is performed.
                If a single integer is provided, it will be unsqueezed to a 1 dimensional shape.
            dtype: The data type to use for the weight and bias parameters.
            eps: :math:`\epsilon` value to prevent division by zero.

        .. code-block:: python
            :linenos:

            layer_norm = tp.LayerNorm(3)

            layer_norm.weight = tp.iota(layer_norm.weight.shape)
            layer_norm.bias = tp.iota(layer_norm.bias.shape)

            input = tp.iota((2, 3), dim=1)
            output = layer_norm(input)

            np_out = cp.from_dlpack(output).get() # doc: omit
            assert np_out.shape == (2, 3)

            torch_tensor = torch.from_dlpack(input) # doc: omit
            torch_ln = torch.nn.LayerNorm(3) # doc: omit
            torch_ln.weight.data = torch.from_dlpack(layer_norm.weight) # doc: omit
            torch_ln.bias.data = torch.from_dlpack(layer_norm.bias) # doc: omit
            assert np.allclose(np_out, cp.from_dlpack(torch_ln(torch_tensor).detach()).get())
        """
        super().__init__()

        self.dtype = dtype

        # Replace with random weights when #74 is completed.
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape

        self.weight = DefaultParameter(normalized_shape, dtype=dtype)

        self.bias = DefaultParameter(normalized_shape, dtype=dtype)

        self.eps = eps

    def forward(self, x: "nvtripy.Tensor") -> "nvtripy.Tensor":
        r"""
        Args:
            x: The input tensor.

        Returns:
            A tensor of the same shape as the input.
        """
        return layernorm(x, self.weight, self.bias, self.eps)
