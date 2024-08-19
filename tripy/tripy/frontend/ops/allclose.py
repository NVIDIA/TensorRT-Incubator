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
from tripy.common.exception import raise_error


@export.public_api(document_under="operations/functions")
def allclose(a: "tripy.Tensor", b: "tripy.Tensor", rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Returns True if the following equation is elementwise True:
    absolute(a - b) <= (atol + rtol * absolute(b))

    Args:
        a: The LHS tensor.
        b: The RHS tensor.
        rtol: The relative tolerance
        atol: The absolute tolerance

    Returns:
        A boolean value

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([1e10,1e-7])
        b = tp.Tensor([1e10,1e-7])
        assert tp.allclose(a, b) == True
    """
    from tripy.frontend.trace.ops.unary_elementwise import abs
    from tripy.frontend.trace.ops.reduce import all
    from tripy.common.datatype import int64, bool as tp_bool

    if a.dtype == int64:
        raise_error("Known issue with i64. Allclose currently does not work with int64 inputs.")
    if a.dtype == tp_bool and b.dtype == tp_bool:
        compare = a == b
    else:
        compare = abs(a - b) <= (atol + rtol * abs(b))
    return bool(all(compare))
