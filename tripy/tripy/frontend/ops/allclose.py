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

from tripy import export, constraints
from tripy.common.exception import raise_error


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int32", "int8"]},
    dtype_constraints={"a": "T1", "b": "T1"},
)
def allclose(a: "tripy.Tensor", b: "tripy.Tensor", rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""
    Returns true if the following equation is true for every element in ``a`` and ``b`` :

    :math:`|a_i - b_i| <= (\text{atol} + \text{rtol} * |b_i|)`

    Args:
        a: First tensor to compare.
        b: Second tensor to compare.
        rtol: The relative tolerance.
        atol: The absolute tolerance.

    Returns:
        ``True`` if the tensors were within the specified tolerances and ``False`` otherwise.

    .. code-block:: python
        :linenos:
        :caption: Within Tolerance

        # doc: print-locals out
        out = tp.allclose(tp.Tensor([1e-7]), tp.Tensor([1.1e-7]))
        assert out

    .. code-block:: python
        :linenos:
        :caption: Outside Tolerance

        # doc: print-locals out
        out = tp.allclose(tp.Tensor([1e-7]), tp.Tensor([1.2e-7]))
        assert not out
    """
    from tripy.frontend.trace.ops.unary_elementwise import abs
    from tripy.frontend.trace.ops.reduce import all

    compare = abs(a - b) <= (atol + rtol * abs(b))
    return bool(all(compare))
