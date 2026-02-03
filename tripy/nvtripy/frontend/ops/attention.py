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

from typing import Optional

from nvtripy import export
from nvtripy.common import datatype
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.attention import Attention
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={
        "query": "T1",
        "key": "T1",
        "value": "T1",
        wrappers.RETURN_VALUE: "T1",
    },
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
)
def attention(
    query: "nvtripy.Tensor",
    key: "nvtripy.Tensor",
    value: "nvtripy.Tensor",
    *,
    mask: Optional["nvtripy.Tensor"] = None,
    normalization_quantize_scale: Optional["nvtripy.Tensor"] = None,
    normalization_operation: str = "softmax",
    causal: bool = False,
    decomposable: bool = False,
    normalization_quantize_to_type: Optional[datatype.dtype] = None,
) -> "nvtripy.Tensor":
    r"""
    Performs a fused multi-head attention operation.

    This operation implements the attention mechanism:

    .. math::
        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

    The operation consists of:

    1. Matrix multiplication between query and transposed key (BMM1)
    2. Optional masking
    3. Normalization (typically softmax)
    4. Optional quantization of the normalized output
    5. Matrix multiplication with value (BMM2)

    Args:
        query: The query tensor with shape ``[batch_size, num_heads_query, sequence_length_query, dim_head]``.
        key: The key tensor with shape ``[batch_size, num_heads_key, sequence_length_key, dim_head]``.
        value: The value tensor with shape ``[batch_size, num_heads_value, sequence_length_value, dim_head]``.
        mask: Optional mask tensor with shape
            ``[batch_size, num_heads_query, sequence_length_query, sequence_length_key]``.
            For boolean masks (dtype=bool), ``True`` indicates positions that are allowed to attend.
            For float masks, the values are added to the attention scores before normalization.
        normalization_quantize_scale: Optional scale tensor for quantizing the normalization output.
            Required if ``normalization_quantize_to_type`` is specified.
        normalization_operation: The normalization operation to use. Must be one of "softmax" or "none".
            Defaults to ``"softmax"``.
        causal: If ``True``, applies causal (autoregressive) masking where each position can only
            attend to earlier positions. Cannot be used together with explicit ``mask``. Defaults to ``False``.
        decomposable: If ``True``, allows the operation to be decomposed into multiple kernels if
            no fused kernel is available. Defaults to ``False``.
        normalization_quantize_to_type: Optional data type for quantizing the normalization output.
            Must be either :class:`nvtripy.float8` or :class:`nvtripy.int8`.
            Requires ``normalization_quantize_scale`` to be provided.

    Returns:
        The attention output tensor with shape ``[batch_size, num_heads_query, sequence_length_query, dim_head]``.

    .. code-block:: python
        :linenos:
        :caption: Basic Attention

        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
        query = tp.iota((batch_size, num_heads, seq_len, head_dim), dtype=tp.float16)
        key = tp.iota((batch_size, num_heads, seq_len, head_dim), dtype=tp.float16)
        value = tp.iota((batch_size, num_heads, seq_len, head_dim), dtype=tp.float16)

        output = tp.attention(query, key, value)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    .. code-block:: python
        :linenos:
        :caption: Attention with Quantization

        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
        query = tp.iota((batch_size, num_heads, seq_len, head_dim), dtype=tp.float16)
        key = tp.iota((batch_size, num_heads, seq_len, head_dim), dtype=tp.float16)
        value = tp.iota((batch_size, num_heads, seq_len, head_dim), dtype=tp.float16)

        # Quantize the normalization output (softmax) to float8
        mask = tp.ones((batch_size, num_heads, seq_len, seq_len), dtype=tp.bool)
        scale = tp.Tensor([1.0], dtype=tp.float16)

        output = tp.attention(query, key, value, mask=mask,
                             normalization_quantize_scale=scale,
                             normalization_quantize_to_type=tp.float8)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    .. code-block:: python
        :linenos:
        :caption: Attention with Mask

        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
        query = tp.iota((batch_size, num_heads, seq_len, head_dim), dtype=tp.float16)
        key = tp.iota((batch_size, num_heads, seq_len, head_dim), dtype=tp.float16)
        value = tp.iota((batch_size, num_heads, seq_len, head_dim), dtype=tp.float16)

        # Create a boolean mask where True indicates positions that can attend
        mask = tp.ones((batch_size, num_heads, seq_len, seq_len), dtype=tp.bool)

        output = tp.attention(query, key, value, mask=mask)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    """
    from nvtripy.common.exception import raise_error

    if normalization_operation not in ("softmax", "none"):
        raise_error(
            f"Invalid normalization operation: {normalization_operation}. Must be one of 'softmax' or 'none'.",
        )

    # Validation checks
    if causal and mask is not None:
        raise_error(
            "Cannot use both `causal` and `mask` at the same time.",
            details=[
                "The `causal` parameter applies implicit causal masking.",
                "Please use either `causal=True` or provide an explicit `mask`.",
            ],
        )

    if normalization_quantize_to_type is not None:
        if normalization_quantize_scale is None:
            raise_error(
                "`normalization_quantize_scale` must be provided when `normalization_quantize_to_type` is specified.",
            )

        if normalization_quantize_to_type not in (datatype.float8, datatype.int8):
            raise_error(
                f"`normalization_quantize_to_type` must be either float8 or int8.",
                details=[f"Got: {normalization_quantize_to_type}"],
            )

    # Collect inputs based on what's provided
    inputs = [query, key, value]
    if mask is not None:
        inputs.append(mask)
    if normalization_quantize_scale is not None:
        inputs.append(normalization_quantize_scale)

    return op_utils.create_op(
        Attention,
        inputs,
        normalization_operation=normalization_operation,
        causal=causal,
        decomposable=decomposable,
        normalization_quantize_to_type=normalization_quantize_to_type,
    )
