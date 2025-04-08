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
from nvtripy.common.datatype import DATA_TYPES
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", "other": "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
)
def equal(input: "nvtripy.Tensor", other: "nvtripy.Tensor") -> bool:
    r"""
    Returns ``True`` if ``input`` and ``other`` have the same shape and elements.

    .. caution:: This function cannot be used in a compiled function or :class:`nvtripy.Module` because it depends on
        evaluating its inputs, which is not allowed during compilation.

    Args:
        input: First tensor to compare.
        other: Second tensor to compare.

    Returns:
        ``True`` if the tensors have the same shape and elements and ``False`` otherwise.

    .. code-block:: python
        :linenos:
        :caption: Identical tensors

        # doc: print-locals a b is_equal
        a = tp.ones((1, 2), dtype=tp.float32)
        b = tp.ones((1, 2), dtype=tp.float32)

        is_equal = tp.equal(a, b)
        assert is_equal

    .. code-block:: python
        :linenos:
        :caption: Different shapes

        # doc: print-locals a b is_equal
        a = tp.ones((1, 2), dtype=tp.float32)
        b = tp.ones((2, 2), dtype=tp.float32)

        is_equal = tp.equal(a, b)
        assert not is_equal

    .. code-block:: python
        :linenos:
        :caption: Different elements

        # doc: print-locals a b is_equal
        a = tp.ones((1, 2), dtype=tp.float32)
        b = tp.zeros((1, 2), dtype=tp.float32)

        is_equal = tp.equal(a, b)
        assert not is_equal
    """
    from nvtripy.frontend.ops.reduce.all import all

    if input.shape != other.shape:
        return False

    return bool(all(input == other))
