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
        super().__init__(data=data, dtype=int32, name=name)

    # Internal use only, leave undocumented so it's not exported.
    # Used to create a DimensionSize with None as the data, which we do not want in the public constructor.
    @staticmethod
    def create_empty(name: Optional[str] = None) -> "tripy.DimensionSize":
        instance = DimensionSize.__new__(DimensionSize)
        super(DimensionSize, instance).__init__(data=None, dtype=int32, name=name)
        return instance

    def __int__(self) -> int:
        return self.tolist()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        val = self.tolist()
        assert isinstance(val, int)
        return str(val)
