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


@export.public_api(document_under="tensor_operations")
def array_equal(a: "tripy.Tensor", b: "tripy.Tensor") -> bool:
    """
    Returns True if the two arrays have the same shape and elements, False otherwise.

    Args:
        a: The LHS tensor.
        b: The RHS tensor.

    Returns:
        A boolean value

    .. code-block:: python
        :linenos:
        :caption: Example

        # doc: print-locals a, b, is_equal

        a = tp.ones((2,2), dtype=tp.int32)
        b = tp.Tensor([
            [1, 1],
            [1, 1]
        ])
        is_equal = tp.array_equal(a, b)
        assert is_equal
    """
    from tripy.frontend.trace.ops.reduce import all
    from tripy.frontend.trace.ops.cast import cast

    # `tp.Shape` overloads `__ne__`, therefore we can directly compare shapes here.
    if a.shape != b.shape:
        return False

    if a.dtype != b.dtype:
        b = cast(b, a.dtype)

    return bool(all(a == b))