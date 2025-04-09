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
from nvtripy.trace.ops.quantize import Quantize
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/quantization")
@wrappers.interface(
    dtype_constraints={"input": "T1", "scale": "T1", "dtype": "T2", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"], "T2": ["int4", "int8", "float8"]},
    convert_to_tensors={"scale"},
)
def quantize(
    input: "nvtripy.Tensor",
    scale: Union["nvtripy.Tensor", numbers.Number, Sequence[numbers.Number], Sequence[Sequence[numbers.Number]]],
    dtype: datatype.dtype,
    dim: Optional[int] = None,
) -> "nvtripy.Tensor":
    """
    Quantizes the input Tensor. The valid quantized data types are
    :class:`nvtripy.int8`, :class:`nvtripy.int4`, :class:`nvtripy.float8`.

    If ``dtype`` is :class:`nvtripy.int4`, the result of this function
    cannot be printed as :class:`nvtripy.int4` is an internal quantized
    data type. It must be dequantized :func:`dequantize` to a higher
    precision first.

    If ``dim`` is not given, this function will perform "per-tensor"
    or "block-wise" quantization.

    * For "per-tensor" quantization, the ``scale`` must be a scalar
      tensor or a single python number.

    * For "block-wise" quantization, the ``dtype`` must only be :class:`nvtripy.int4`.
      The ``input`` tensor must only have 2 dimensions, e.g. ``[D0, D1]``.
      The ``scale`` must also be a 2-D tensor or a 2-D python sequence.
      The first dimension of ``scale`` must be able to divide ``D0``,
      where "blocking" is performed. The second dimension of ``scale``
      must equal to ``D1``.

    If ``dim`` is given, this function will perform "per-channel"
    quantization. The ``scale`` must be a 1-D tensor or a python sequence
    both with size of ``input.shape[dim]``.

    Args:
        input: The input tensor.
        scale: The scale tensor. Must be a constant tensor.
        dtype: The quantization data type. Must be a valid quantized data type (see above).
        dim: The dimension for per-channel quantization

    Returns:
        Quantized Tensor.

    .. code-block:: python
        :linenos:
        :caption: Per-tensor quantization

        input = tp.reshape(tp.arange(6, tp.float32), (2, 3))
        scale = 0.99872
        # output = tp.quantize(input, scale, tp.int8)

        # expected = (np.reshape(np.arange(6, dtype=np.float32), (2, 3)) / scale).astype(np.int8) # doc: omit
        # assert np.array_equal(cp.from_dlpack(output).get(), expected)

    .. code-block:: python
        :linenos:
        :caption: Per-channel quantization

        input = tp.Tensor([[0, 1, 2], [3, 4, 5]], dtype=tp.float32)
        scale = [0.99872, 0.96125]
        output = tp.quantize(input, scale, tp.int8, dim=0)

        expected = (np.reshape(np.arange(6, dtype=np.float32), (2, 3)) / np.array(scale).reshape(2, 1)).astype(np.int8) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), expected)

    .. code-block:: python
        :linenos:
        :caption: Block-wise quantization

        # doc: print-locals input, output

        input = tp.Tensor([[0, 1], [2, 3]], dtype=tp.float32)
        scale = [[1.0, 1.0]]
        quant = tp.quantize(input, scale, tp.int4)
        output = tp.dequantize(quant, scale, tp.float32)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[0, 1], [2, 3]], dtype=np.float32))

    .. seealso:: :func:`dequantize`
    """
    op_utils.check_qdq_args(input, scale, dtype, dim, True)

    return op_utils.create_op(Quantize, [input, scale], dtype, dim)
