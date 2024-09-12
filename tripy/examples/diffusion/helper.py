import math
from functools import reduce
from typing import List, Callable, Optional

import tripy as tp


def scaled_dot_product_attention(
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    embedding_dim: Optional[int] = None,
    attn_mask: Optional[tp.Tensor] = None,
    is_causal: bool = False,
    dtype: tp.dtype = tp.float32
) -> tp.Tensor:
    """
    Computes scaled dot-product attention.
    `self` is the query tensor, `key` is the key tensor, and `value` is the value tensor.

    - Described: https://paperswithcode.com/method/scaled
    - Paper: https://arxiv.org/abs/1706.03762v7
    """
    if is_causal:  # this path is not called in demoDiffusion
        target_shape = query.shape[-2:-1] + key.shape[-2:-1]
        # TODO: #228: WAR to prevent computing output rank in infer_rank for reshape
        target_shape.trace_tensor.shape = (2,)
        attn_mask = tp.cast(tp.tril(tp.ones(target_shape)), tp.bool)
    if attn_mask is not None and attn_mask.dtype == tp.bool:
        attn_mask = tp.where((attn_mask == 0), tp.ones_like(attn_mask, dtype=dtype) * -float("inf"), tp.zeros_like(attn_mask, dtype=dtype))
    if attn_mask is not None:
        attn_mask = tp.cast(attn_mask, dtype)
    qk = query @ tp.transpose(key, -2, -1) / math.sqrt(embedding_dim)
    return tp.cast(tp.softmax((qk + attn_mask) if attn_mask is not None else qk, -1), query.dtype) @ value


def sequential(input: tp.Tensor, ll: List[Callable[[tp.Tensor], tp.Tensor]]):
    """
    Applies a sequence of functions to `self` chaining the output of each function to the input of the next.
    """
    return reduce(lambda x, f: f(x), ll, input)


def clamp(tensor: tp.Tensor, min: int, max: int):
    return tp.minimum(tp.maximum(tensor, tp.ones_like(tensor) * min), tp.ones_like(tensor) * max)