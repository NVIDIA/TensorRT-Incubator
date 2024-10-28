#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

from tripy import export
from tripy.common.datatype import int32
from tripy.frontend.tensor import Tensor


@export.public_api(document_under="tensor")
class DimensionSize(Tensor):
    """
    A 0D, :class:`int32` tensor that represents a scalar value extracted from the shape of a tensor.
    """

    def __init__(self, data: int, name: Optional[str] = None) -> None:
        r"""
        Args:
            data: The value of the DimensionSize, which should be a scalar integer.
            name: An optional name.
        """

        from tripy.common.exception import raise_error

        # This branch of the constructor is basically a backdoor for `.shape` to work.
        # It is not generally possible to cast Tensors to DimensionSizes.
        if isinstance(data, Tensor):
            # these fields can be None in the case of an uninitialized tensor (like Tensor(None))
            if data.trace_tensor.rank is not None and data.trace_tensor.rank != 0:
                raise_error(
                    f"Scalar shape tensors must be of rank 0, but input tensor is rank {data.rank}", details=[data]
                )
            if data.dtype is not None and data.dtype != int32:
                raise_error(
                    f"Scalar shape tensor must have int32 member, but input tensor has data type {data.dtype}",
                    details=[data],
                )

            # the shape of data should correspond to the given rank
            super().__init__(data=None, dtype=int32, name=name, device=data.device)
            # share the underlying data
            self.trace_tensor = data.trace_tensor
            self.stack_info = data.stack_info
        else:
            # NOTE: We do not use isinstance here because bool is a subclass of int.
            assert data is None or type(data) is int, "DimensionSize can only be created from integers"
            super().__init__(data=data, dtype=int32, name=name)

    def __int__(self) -> int:
        return self.tolist()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        val = self.tolist()
        assert isinstance(val, int)
        return str(val)
