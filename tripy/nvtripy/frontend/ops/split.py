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

from typing import Sequence, Union

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.trace.ops.split import Split
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
)
def split(
    input: "nvtripy.Tensor", indices_or_sections: Union[int, Sequence[int]], dim: int = 0
) -> Union["nvtripy.Tensor", Sequence["nvtripy.Tensor"]]:
    r"""
    Splits `input` along the dimension `dim`, producing slices of the `input` tensor.

    If given a single `int` for `indices_or_sections` (let us call it :math:`n`),
    it produces :math:`n` slices of equal size along dimension `dim` as long as :math:`n` divides the size of dimension `dim`.
    For example, if `input` is one-dimensional and the size of dimension `dim` is :math:`k`, then the result is
    :math:`\texttt{input[:} k/n \texttt{]}`, :math:`\texttt{input[} k/n \texttt{:} 2k/n \texttt{]}`,
    :math:`\ldots`, :math:`\texttt{input[} (n-1)k/n \texttt{:]}`.

    If given a sequence of values for `indices_or_sections`, these will be treated as indices for creating slices.
    For example, if we call the indices :math:`i_0`, :math:`i_1`, :math:`\ldots`, :math:`i_n` and assume `input` is one-dimensional,
    the result is equivalent to :math:`input[:i_0]`, :math:`input[i_0:i_1]`, :math:`input[i_1:i_2]`, :math:`\ldots`, :math:`input[i_n:]`.

    Args:
        input: The input tensor.

        indices_or_sections: If a single integer, it gives the number of equal slices to produce.
            If a list of integers, it gives boundary indices for the slices.

        dim: The dimension along which the slices are done. All other dimensions are included in full.

    Returns:
        A list of slices per the above specification or a single tensor if only one slice is created.

    .. code-block:: python
        :linenos:
        :caption: Simple case.

        input = tp.reshape(tp.arange(16, dtype=tp.float32), (4, 4))
        outputs = tp.split(input, 2, dim=0)
        assert np.array_equal(cp.from_dlpack(outputs[0]).get(), cp.from_dlpack(input[:2, :]).get())
        assert np.array_equal(cp.from_dlpack(outputs[1]).get(), cp.from_dlpack(input[2:, :]).get())

    .. code-block:: python
        :linenos:
        :caption: Choosing a different dimension.

        input = tp.reshape(tp.arange(16, dtype=tp.float32), (4, 4))
        outputs = tp.split(input, 2, dim=1)
        assert np.array_equal(cp.from_dlpack(outputs[0]).get(), cp.from_dlpack(input[:, :2]).get())
        assert np.array_equal(cp.from_dlpack(outputs[1]).get(), cp.from_dlpack(input[:, 2:]).get())

    .. code-block:: python
        :linenos:
        :caption: Multiple index arguments.

        input = tp.reshape(tp.arange(16, dtype=tp.float32), (4, 4))
        outputs = tp.split(input, [1, 2])
        assert np.array_equal(cp.from_dlpack(outputs[0]).get(), cp.from_dlpack(input[:1, :]).get())
        assert np.array_equal(cp.from_dlpack(outputs[1]).get(), cp.from_dlpack(input[1:2, :]).get())
        assert np.array_equal(cp.from_dlpack(outputs[2]).get(), cp.from_dlpack(input[2:, :]).get())
    """
    if dim < 0 or dim >= input.rank:
        raise_error(f"Invalid split dimension {dim}", details=[input])
    if isinstance(indices_or_sections, int):
        if indices_or_sections <= 0:
            raise_error(f"Number of sections argument must be positive, but given {indices_or_sections}")
        num_outputs = indices_or_sections
    else:
        if not indices_or_sections:
            raise_error("Split indices must not be empty")
        last = None
        for index in indices_or_sections:
            if last and index < last:
                raise_error(f"Split indices must be given in ascending order, but given {indices_or_sections}")
            last = index
        num_outputs = len(indices_or_sections) + 1  # add 1 because of the last split

    return Split.build(inputs=[input], indices_or_sections=indices_or_sections, dim=dim, num_outputs=num_outputs)
