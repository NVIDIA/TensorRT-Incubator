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

import numbers
from typing import Optional, Sequence, Union

from nvtripy import export
from nvtripy.common import datatype
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.dequantize import Dequantize
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/quantization")
@wrappers.interface(
    dtype_constraints={"input": "T1", "scale": "T2", "dtype": "T2", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={"T1": ["int4", "int8", "float8"], "T2": ["float32", "float16", "bfloat16"]},
    convert_to_tensors={"scale"},
)
def dequantize(
    input: "nvtripy.Tensor",
    scale: Union["nvtripy.Tensor", numbers.Number, Sequence[numbers.Number], Sequence[Sequence[numbers.Number]]],
    dtype: datatype.dtype,
    dim: Optional[int] = None,
) -> "nvtripy.Tensor":
    """
    Dequantizes the input tensor.

    If ``dim`` is not given, this function will perform "per-tensor"
    or "block-wise" dequantization.

    * For "per-tensor" dequantization, the ``scale`` must be a scalar
      tensor or a single python number.

    * For "block-wise" dequantization, the ``dtype`` must only be :class:`nvtripy.int4`.
      The ``input`` tensor must only have 2 dimensions, e.g. ``[D0, D1]``.
      The ``scale`` must also be a 2-D tensor or a 2-D python sequence.
      The first dimension of ``scale`` must be able to divide ``D0``,
      where "blocking" is performed. The second dimension of ``scale``
      must equal to ``D1``.


    If ``dim`` is given, this function will perform "per-channel"
    dequantization. The ``scale`` must be a 1-D tensor or a python sequence
    both with size of ``input.shape[dim]``.

    Args:
        input: The input tensor with a valid quantized data type.
        scale: The scale tensor. Must be a constant tensor.
        dtype: The data type after dequantization. Must be :class:`nvtripy.float32` or :class:`nvtripy.float16`.
        dim: The dimension for per-channel dequantization

    Returns:
        The dequantized tensor.

    .. code-block:: python
        :linenos:
        :caption: Per-tensor dequantization

        input = tp.Tensor([1, 2, 3], dtype=tp.int8)
        scale = 0.99872
        output = tp.dequantize(input, scale, tp.float32)

        expected = (np.array([1, 2, 3], dtype=np.int8) * scale).astype(np.float32) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), expected)

    .. code-block:: python
        :linenos:
        :caption: Per-channel dequantization

        input = tp.Tensor([[1, 2, 3], [4, 5, 6]], dtype=tp.int8)
        scale = [0.99872, 0.96125]
        output = tp.dequantize(input, scale, tp.float32, dim=0)

        expected = (np.array([[1, 2, 3], [4, 5, 6]]) * np.array(scale).reshape(2, 1)).astype(np.float32) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), expected)

    .. code-block:: python
        :linenos:
        :caption: Block-wise dequantization

        # doc: print-locals input, output

        input = tp.Tensor([[0, 1], [2, 3]], dtype=tp.float32)
        scale = [[1.0, 1.0]]
        quant = tp.quantize(input, scale, tp.int4)
        output = tp.dequantize(quant, scale, tp.float32)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[0, 1], [2, 3]], dtype=np.float32))

    .. seealso:: :func:`quantize`
    """
    op_utils.check_qdq_args(input, scale, dtype, dim, False)

    return op_utils.create_op(Dequantize, [input, scale], dtype, dim)
