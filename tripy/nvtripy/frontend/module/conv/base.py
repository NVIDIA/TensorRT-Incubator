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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from nvtripy import utils
from nvtripy.common import datatype
from nvtripy.common.exception import raise_error
from nvtripy.frontend.module.module import Module
from nvtripy.frontend.module.parameter import DefaultParameter
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.tensor import Tensor


@dataclass
@utils.wrappers.constant_fields(["dtype", "padding", "stride", "groups", "dilation"])
class ConvBase(Module):
    r"""Base class for sharing common functionality between Conv and ConvTranspose."""

    dtype: datatype.dtype
    padding: Sequence[Sequence[int]]
    stride: Sequence[int]
    groups: int
    dilation: Sequence[int]
    bias: Optional[Tensor]

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

        super().__init__()

        self.groups = utils.utils.default(groups, 1)

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

        num_spatial = len(kernel_dims)
        op_utils.check_conv_pooling_args(kernel_dims, stride, padding, dilation)
        self.padding = utils.utils.default(padding, tuple(((0, 0) for _ in range((num_spatial)))))
        self.stride = utils.utils.default(stride, (1,) * num_spatial)
        self.dilation = utils.utils.default(dilation, (1,) * num_spatial)

        self.bias = None
        if bias:
            bias_shape = (out_channels,)
            self.bias = DefaultParameter(bias_shape, dtype=dtype)

        self.dtype = dtype
