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

from typing import Optional

import math
import tripy as tp
from tripy.common.datatype import float32


def scaled_dot_product_attention(
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    embedding_dim: Optional[int] = None,
    attn_mask: Optional[tp.Tensor] = None,
    is_causal: bool = False,
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
        attn_mask = tp.where(
            (attn_mask == 0),
            tp.ones_like(attn_mask) * -float("inf"),
            tp.zeros_like(attn_mask),
        )
    qk = query @ tp.transpose(key, -2, -1) / tp.sqrt(tp.cast(embedding_dim, tp.float32))
    return (
        tp.cast(
            tp.softmax((qk + attn_mask) if attn_mask is not None else qk, -1),
            query.dtype,
        )
        @ value
    )


class MLP(tp.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: tp.Module = tp.relu,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = []
        for n, k in zip([input_dim] + h, h + [output_dim]):
            self.layers.append(tp.Linear(n, k))

        self.sigmoid_output = sigmoid_output
        self.act = activation

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = tp.sigmoid(x)
        return x


class LayerNorm2d(tp.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        from tripy.frontend.module.parameter import DefaultParameter

        self.weight = DefaultParameter((num_channels,), float32)
        self.bias = DefaultParameter((num_channels,), float32)
        self.eps = eps

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        u = tp.mean(x, dim=1, keepdim=True)
        s = tp.mean((x - u) ** 2, dim=1, keepdim=True)
        x = (x - u) / tp.sqrt(s + self.eps)
        w = tp.unsqueeze(tp.unsqueeze(self.weight, 1), 2)
        b = tp.unsqueeze(tp.unsqueeze(self.bias, 1), 2)
        x = w * x + b
        return x
