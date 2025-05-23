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

from nvtripy import constants, export, utils
from nvtripy.common import datatype
from nvtripy.common.exception import raise_error
from nvtripy.frontend.module.module import Module
from nvtripy.frontend.module.parameter import DefaultParameter
from nvtripy.frontend.tensor import Tensor

from nvtripy.frontend.ops import utils as op_utils
from nvtripy.utils import wrappers
from nvtripy.trace.ops.instancenorm import InstanceNorm as InstanceNormOp


@wrappers.interface(
    dtype_constraints={"input": "T1", "weight": "T1", "bias": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def instancenorm(
    input: "nvtripy.Tensor",
    weight: "nvtripy.Tensor",
    bias: "nvtripy.Tensor",
    num_channels: int,
    eps: float,
) -> "nvtripy.Tensor":

    input_rank = input.rank

    if input_rank < 3:
        raise_error(
            f"Input must have a rank of at least 3, but got input of rank: {input.rank}",
            details=[
                "Input is expected to have shape (N, C, D1, ...) where N is the batch size, C is the number of channels, and D1, ... are the spatial dimensions"
            ],
        )

    if input.trace_tensor.shape[1] != constants.DYNAMIC_DIM and input.trace_tensor.shape[1] != num_channels:
        raise_error(
            f"Expected {num_channels} channels in the input, but got {input.shape[1]} channels",
            details=[
                "The input channel dimension must match the number of channels specified to the InstanceNorm module."
            ],
        )

    # TensorRT expects weight & bias to have shape [1, C, 1, 1, ...]
    from nvtripy.frontend.ops.reshape import reshape

    broadcast_shape = (1, num_channels) + (1,) * (input_rank - 2)
    weight = reshape(weight, broadcast_shape)
    bias = reshape(bias, broadcast_shape)

    # The MLIR Graph may have dynamic dimensions, so we explicitly set static dimensions in the trace shape tensors
    weight.trace_tensor.shape = broadcast_shape
    bias.trace_tensor.shape = broadcast_shape
    input.trace_tensor.shape = input.trace_tensor.shape[:1] + (num_channels,) + input.trace_tensor.shape[2:]

    return op_utils.create_op(
        InstanceNormOp,
        [input, weight, bias],
        num_channels=num_channels,
        eps=eps,
    )


@export.public_api(document_under="operations/modules")
@dataclass
@utils.wrappers.constant_fields(["num_channels", "dtype", "eps"])
class InstanceNorm(Module):
    r"""
    Applies Instance Normalization over a mini-batch of inputs:

    :math:`\text{InstanceNorm}(x) = \Large \frac{x - \mu}{ \sqrt{\sigma^2 + \epsilon}} \normalsize * \gamma + \beta`

    where :math:`\mu` is the mean and :math:`\sigma^2` is the variance, computed per channel
    for each instance in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameters
    of shape (C).

    InstanceNorm is similar to LayerNorm, but statistics are computed per channel across spatial dimensions,
    whereas LayerNorm is computed across all dimensions of a sample.
    """

    num_channels: int
    r"""Number of channels/features expected in the input."""

    dtype: datatype.dtype
    r"""The data type used to perform the operation."""

    weight: Tensor
    r"""The :math:`\gamma` parameter of shape (num_channels)."""

    bias: Tensor
    r"""The :math:`\beta` parameter of shape (num_channels)."""

    eps: float
    r"""A value added to the denominator for numerical stability."""

    def __init__(
        self,
        num_channels: int,
        dtype: datatype.dtype = datatype.float32,
        eps: float = 1e-5,
    ) -> None:
        r"""
        Args:
            num_channels: Number of channels/features expected in the input
            dtype: The data type to use for the module parameters
            eps: The epsilon value added to the denominator for numerical stability

        .. code-block:: python
            :linenos:

            instance_norm = tp.InstanceNorm(3)
            instance_norm.weight = tp.ones((3,))
            instance_norm.bias = tp.zeros((3,))

            input_tensor = tp.ones((2, 3, 4, 4))
            output = instance_norm(input_tensor)

            np_out = cp.from_dlpack(output).get() # doc: omit
            assert np_out.shape == (2, 3, 4, 4)

            torch_tensor = torch.from_dlpack(input_tensor) # doc: omit
            torch_in = torch.nn.InstanceNorm2d(3, affine=True) # doc: omit
            torch_in.weight.data = torch.from_dlpack(instance_norm.weight) # doc: omit
            torch_in.bias.data = torch.from_dlpack(instance_norm.bias) # doc: omit
            torch_output = torch_in(torch_tensor) # doc: omit
            assert np.allclose(np_out, torch_output.detach().cpu().numpy())
        """
        super().__init__()

        self.num_channels = num_channels
        self.dtype = dtype
        self.eps = eps
        self.weight = DefaultParameter((num_channels,), dtype=dtype)
        self.bias = DefaultParameter((num_channels,), dtype=dtype)

    def forward(self, x: "nvtripy.Tensor") -> "nvtripy.Tensor":
        r"""
        Args:
            x: Input tensor of shape [N, C, ...] where C is the number of features

        Returns:
            Normalized tensor of the same shape as input
        """
        return instancenorm(x, self.weight, self.bias, self.num_channels, self.eps)
