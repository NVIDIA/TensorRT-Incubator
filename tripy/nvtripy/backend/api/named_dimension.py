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
from nvtripy import export


@export.public_api(document_under="compiling_code/input_info")
class NamedDimension:
    """
    Represents a named dimension with its shape bounds.

    Two dimensions with the same name must be equal at runtime.
    This equality can be exploited by the compiler to achieve better optimizations.
    """

    def __init__(self, name: str, min: int, opt: int, max: int) -> None:
        """
        Args:
            name: The name of the dimension.
            min: The minimum size of the dimension.
            opt: The size of the dimension for which the compiler should optimize.
            max: The maximum size of the dimension.

        .. code-block:: python
            :linenos:
            :caption: Creating a Named Dimension

            batch = tp.NamedDimension("batch", 1, 2, 3)
            assert batch.bounds == (1, 2, 3)
        """
        self.bounds = (min, opt, max)
        self.name = name

    def __str__(self) -> str:
        return f"NamedDimension<name: '{self.name}', bounds: {self.bounds}>"
