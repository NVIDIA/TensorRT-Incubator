import math
from typing import Optional

import nvtripy as tp


def scaled_dot_product_attention(
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    embedding_dim: int,
    attn_mask: Optional[tp.Tensor] = None,
) -> tp.Tensor:
    dtype = query.dtype
    if attn_mask is not None and attn_mask.dtype == tp.bool:
        attn_mask = tp.where(
            (attn_mask == 0),
            tp.ones_like(attn_mask, dtype=dtype) * -float("inf"),
            tp.zeros_like(attn_mask, dtype=dtype),
        )
    if attn_mask is not None:
        attn_mask = tp.cast(attn_mask, dtype)
    k_t = tp.transpose(key, -2, -1)
    qk = (query @ k_t) * (1.0 / math.sqrt(embedding_dim))
    return tp.softmax((qk + attn_mask) if attn_mask is not None else qk, -1) @ value


def clamp(tensor: tp.Tensor, min: int, max: int):
    return tp.minimum(tp.maximum(tensor, tp.ones_like(tensor) * min), tp.ones_like(tensor) * max)
