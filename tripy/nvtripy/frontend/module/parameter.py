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

from typing import Sequence, Optional
from nvtripy.utils.stack_info import get_stack_info


class ParameterBase:
    def __init__(self, shape: Optional[Sequence[int]], dtype: "nvtripy.dtype") -> None:
        self.shape = shape
        self.dtype = dtype
        self.stack_info = get_stack_info(1)


class DefaultParameter(ParameterBase):
    """
    Denotes a parameter in a module and its expected shape and data type.

    Must be replaced with real weights before the module can be run.
    """

    pass


class OptionalParameter(ParameterBase):
    """
    Denotes an optional parameter in a module and its expected shape and data type.
    """

    pass
