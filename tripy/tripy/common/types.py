#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tripy import export


@export.public_api()
class TensorLiteral:
    """
    This class is a type annotation for tensor literals.
    A tensor literal can be a Python number or a sequence of tensor literals
    (i.e., a sequence of numbers of any depth).

    Equivalent to this recursive type definition:
    ```
    TensorLiteral = Union[numbers.Number, Sequence["TensorLiteral"]]
    ```
    """

    # This is a WAR to avoid implementing general support for recursive type annotations in the registry.
    # In the above example, the "TensorLiteral" would be parsed as typing.ForwardRef and would be more challenging
    # to handle correctly in the registry, so this workaround is likely preferable.
    pass
