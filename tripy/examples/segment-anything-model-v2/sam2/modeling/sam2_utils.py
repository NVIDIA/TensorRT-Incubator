# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling SAM2 with Tripy or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
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

from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import nvtripy as tp


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


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

    if is_causal:
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
    if embedding_dim is None:
        embedding_dim = query.shape[-1]
    qk = query @ tp.transpose(key, -2, -1) / tp.sqrt(tp.cast(embedding_dim, query.dtype))
    return (
        tp.cast(
            tp.softmax((qk + attn_mask) if attn_mask is not None else qk, -1),
            query.dtype,
        )
        @ value
    )


class TorchMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MLP(tp.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: Callable[[tp.Tensor], tp.Tensor] = tp.relu,
        sigmoid_output: bool = False,
        dtype=tp.float32,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = []
        for n, k in zip([input_dim] + h, h + [output_dim]):
            self.layers.append(tp.Linear(n, k, dtype=dtype))

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
        from nvtripy.frontend.module.parameter import DefaultParameter

        self.weight = DefaultParameter((num_channels,), tp.float32)
        self.bias = DefaultParameter((num_channels,), tp.float32)
        self.eps = eps

    def __call__(self, x: tp.Tensor) -> tp.Tensor:
        original_dtype = x.dtype
        x = tp.cast(x, tp.float32)
        u = tp.mean(x, dim=1, keepdim=True)
        s = tp.mean((x - u) ** 2, dim=1, keepdim=True)
        x = (x - u) / tp.sqrt(s + self.eps)
        w = tp.unsqueeze(tp.unsqueeze(self.weight, 1), 2)
        b = tp.unsqueeze(tp.unsqueeze(self.bias, 1), 2)
        x = w * x + b
        x = tp.cast(x, original_dtype)
        return x


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return tp.relu
    if activation == "gelu":
        return tp.gelu
    if activation == "glu":
        return tp.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def cartesian_via_polar(abs, angles):
    r"""
    Constructs the real-valued cartesian coordinates from magnitude and angle representing polar coordinates. For input
    ``abs`` and ``angles`` of shape :math:`(m_1, m_2, \ldots, m_i),` this function returns a new real tensor of shape
    """
    real = abs * tp.cos(angles)
    imag = abs * tp.sin(angles)
    return tp.stack([real, imag], dim=-1)


def mul_as_complex(tensor1, tensor2):
    r"""
    Multiplies two tensors (elementwise) as if they were complex-valued.
    The last dimension for both tensors must be 2, representing the real and imaginary components.
    """
    flattened1 = tensor1
    flattened2 = tensor2

    real = flattened1[:, :, :, :, 0] * flattened2[:, :, :, :, 0] - flattened1[:, :, :, :, 1] * flattened2[:, :, :, :, 1]
    imag = flattened1[:, :, :, :, 0] * flattened2[:, :, :, :, 1] + flattened1[:, :, :, :, 1] * flattened2[:, :, :, :, 0]
    return tp.stack([real, imag], dim=-1)
