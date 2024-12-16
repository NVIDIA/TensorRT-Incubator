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

from nvtripy import export, wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", "other": "T1"}, dtype_variables={"T1": ["float32", "float16", "bfloat16"]}
)
def allclose(input: "nvtripy.Tensor", other: "nvtripy.Tensor", rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""
    Returns ``True`` if the following equation is true for every element in ``input`` and ``other`` :

    :math:`|\text{input}_i - \text{other}_i| <= (\text{atol} + \text{rtol} * |\text{other}_i|)`

    Args:
        input: First tensor to compare.
        other: Second tensor to compare.
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
    from nvtripy.frontend.trace.ops.reduce import all
    from nvtripy.frontend.trace.ops.unary_elementwise import abs

    compare = abs(input - other) <= (atol + rtol * abs(other))
    return bool(all(compare))
