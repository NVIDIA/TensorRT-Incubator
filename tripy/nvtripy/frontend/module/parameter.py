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

import math
from typing import Optional, Sequence

from nvtripy.frontend.tensor import Tensor


class DefaultParameter(Tensor):
    """
    Behaves exactly like a tensor except does not cause
    data to be materialized for shape/dtype checks.
    Useful for initializing module parameters.
    """

    def __init__(self, shape: Optional[Sequence[int]], dtype: "nvtripy.dtype") -> None:
        from nvtripy.frontend.ops.tensor_initializers import arange
        from nvtripy.frontend.trace.ops.reshape import reshape

        is_shape_known = True
        if shape is None:
            is_shape_known = False
            shape = []
        tensor = reshape(arange(math.prod(shape), dtype), shape)
        self.__dict__ = tensor.__dict__

        self._shape = list(shape)
        self._dtype = dtype
        self.is_shape_known = is_shape_known

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype
