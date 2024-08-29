#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
from typing import List, Tuple

import tripy as tp
from dataclasses import dataclass

from examples.diffusion.helper import scaled_dot_product_attention, sequential
from examples.diffusion.vae_model import Upsample, Downsample

@dataclass
class UNet15Config:
    io_channels: int = 4
    model_channels: int = 320
    channel_mult: Tuple[int] = (1, 2, 4, 4)
    attention_resolutions: Tuple[int] = (4, 2, 1)
    num_heads: int = 8
    context_dim: int = 768
    emb_channels: int = 1280


# Used for UNet, not to be confused with ResnetBlock, called ResnetBlock2D in HF diffusers
class ResBlock(tp.Module):
    def __init__(self, channels, emb_channels, out_channels):
        self.norm1 = tp.GroupNorm(32, channels)
        self.conv1 = tp.Conv(channels, out_channels, (3, 3), padding=((1, 1), (1, 1)))
        self.time_emb_proj = tp.Linear(emb_channels, out_channels)
        self.norm2 = tp.GroupNorm(32, out_channels)
        self.conv2 = tp.Conv(out_channels, out_channels, (3, 3), padding=((1, 1), (1, 1)))
        self.nonlinearity = tp.silu
        self.conv_shortcut = tp.Conv(channels, out_channels, (1, 1)) if channels != out_channels else lambda x: x

    def __call__(self, x, emb):
        h = self.conv1(self.nonlinearity(self.norm1(x)))
        emb_out = self.time_emb_proj(self.nonlinearity(emb))
        target_shape = emb_out.shape + (1, 1)
        # TODO: #228: WAR to prevent computing output rank in infer_rank for reshape
        target_shape.trace_tensor.shape = (emb_out.rank + 2,)
        h = h + tp.reshape(emb_out, target_shape)
        h = self.conv2(self.nonlinearity(self.norm2(h)))
        ret = self.conv_shortcut(x) + h
        return ret


class CrossAttention(tp.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        self.to_q = tp.Linear(query_dim, n_heads * d_head, bias=False)
        self.to_k = tp.Linear(context_dim, n_heads * d_head, bias=False)
        self.to_v = tp.Linear(context_dim, n_heads * d_head, bias=False)
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = [tp.Linear(n_heads * d_head, query_dim)]

    def __call__(self, x, context=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = [
            tp.transpose(tp.reshape(y, (x.shape[0], -1, self.num_heads, self.head_size)), 1, 2) for y in (q, k, v)
        ]
        attention = tp.transpose(scaled_dot_product_attention(q, k, v, embedding_dim=self.head_size), 1, 2)
        h_ = tp.reshape(attention, (x.shape[0], -1, self.num_heads * self.head_size))
        out = sequential(h_, self.to_out)
        return out


class GEGLU(tp.Module):
    def __init__(self, dim_in, dim_out):
        self.proj = tp.Linear(dim_in, dim_out * 2)
        self.dim_out = dim_out

    def __call__(self, x):
        proj = self.proj(x)
        x, gate = tp.split(proj, 2, proj.rank - 1)  # TODO: allow negative dim in split
        return x * tp.gelu(gate)


class Dummy(tp.Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class FeedForward(tp.Module):
    def __init__(self, dim, mult=4):
        self.net = [
            GEGLU(dim, dim * mult),
            Dummy(),  # Accounts for Dropout layer, needed for weight loading
            tp.Linear(dim * mult, dim),
        ]

    def __call__(self, x):
        return sequential(x, self.net)


class BasicTransformerBlock(tp.Module):
    def __init__(self, dim, context_dim, n_heads, d_head):
        self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head)
        self.norm1 = tp.LayerNorm(dim)
        self.norm2 = tp.LayerNorm(dim)
        self.norm3 = tp.LayerNorm(dim)

    def __call__(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(tp.Module):  # Transformer2dModel in HF diffusers
    def __init__(self, channels, context_dim, n_heads, d_head):
        self.norm = tp.GroupNorm(32, channels)
        assert channels == n_heads * d_head
        self.proj_in = tp.Conv(channels, n_heads * d_head, (1, 1))
        self.transformer_blocks = [BasicTransformerBlock(channels, context_dim, n_heads, d_head)]
        self.proj_out = tp.Conv(n_heads * d_head, channels, (1, 1))

    def __call__(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = tp.permute(tp.reshape(x, (b, c, h * w)), (0, 2, 1))
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = tp.reshape(tp.permute(x, (0, 2, 1)), (b, c, h, w))
        ret = self.proj_out(x) + x_in
        return ret

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = tp.exp(-math.log(max_period) * tp.arange(half) / half)
    args = timesteps * freqs
    return tp.reshape(tp.concatenate([tp.cos(args), tp.sin(args)], dim=0), (1, -1))


class TimestepEmbedding(tp.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        self.linear_1 = tp.Linear(in_channels, time_embed_dim)
        self.act = tp.silu
        self.linear_2 = tp.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class CrossAttnDownBlock2D(tp.Module):
    def __init__(self, start_channels, channels, n_heads, d_head, context_dim=768, emb_channels=1280):
        self.attentions = [
            SpatialTransformer(channels, context_dim, n_heads, d_head),
            SpatialTransformer(channels, context_dim, n_heads, d_head),
        ]
        self.resnets = [ResBlock(start_channels, emb_channels, channels), ResBlock(channels, emb_channels, channels)]
        self.downsamplers = [Downsample(channels)]

    def __call__(self, x, emb, context):
        one = self.resnets[0](x, emb)
        two = self.attentions[0](one, context)
        three = self.resnets[1](two, emb)
        four = self.attentions[1](three, context)
        five = self.downsamplers[0](four)
        return five, [two, four, five]  # saved inputs


class DownBlock2D(tp.Module):
    def __init__(self, channels, emb_channels=1280):
        self.resnets = [ResBlock(channels, emb_channels, channels), ResBlock(channels, emb_channels, channels)]

    def __call__(self, x, emb):
        temp = self.resnets[0](x, emb)
        out = self.resnets[1](temp, emb)
        return out, [temp, out]


class UNetMidBlock2DCrossAttn(tp.Module):
    def __init__(self, channels, n_heads, d_head, context_dim=768, emb_channels=1280):
        self.attentions = [SpatialTransformer(channels, context_dim, n_heads, d_head)]
        self.resnets = [ResBlock(channels, emb_channels, channels), ResBlock(channels, emb_channels, channels)]

    def __call__(self, x, emb, context):
        x = self.resnets[0](x, emb)
        x = self.attentions[0](x, context)
        x = self.resnets[1](x, emb)
        return x


class UpBlock2D(tp.Module):
    def __init__(self, channels, out_channels, emb_channels=1280):
        self.resnets = [
            ResBlock(channels, emb_channels, out_channels),
            ResBlock(channels, emb_channels, out_channels),
            ResBlock(channels, emb_channels, out_channels),
        ]
        self.upsamplers = [Upsample(out_channels)]

    def __call__(self, x, emb, saved_inputs):
        for resblock in self.resnets:
            x = tp.concatenate([x, saved_inputs.pop()], dim=1)
            x = resblock(x, emb)
        return self.upsamplers[0](x)


class CrossAttnUpBlock2D(tp.Module):
    def __init__(
        self,
        start_channels: List[int],
        channels,
        n_heads,
        d_head,
        context_dim=768,
        emb_channels=1280,
        use_upsampler=True,
    ):
        assert len(start_channels) == 3, "Must pass in the start channels for all three resblocks separately"
        self.attentions = [
            SpatialTransformer(channels, context_dim, n_heads, d_head),
            SpatialTransformer(channels, context_dim, n_heads, d_head),
            SpatialTransformer(channels, context_dim, n_heads, d_head),
        ]
        self.resnets = [
            ResBlock(start_channels[0], emb_channels, channels),
            ResBlock(start_channels[1], emb_channels, channels),
            ResBlock(start_channels[2], emb_channels, channels),
        ]
        if use_upsampler:
            self.upsamplers = [Upsample(channels)]

    def __call__(self, x, emb, context, saved_inputs):
        for i in range(len(self.attentions)):
            x = tp.concatenate([x, saved_inputs.pop()], dim=1)
            x = self.resnets[i](x, emb)
            x = self.attentions[i](x, context)
        if hasattr(self, "upsamplers"):
            x = self.upsamplers[0](x)
        return x


class UNetModel(tp.Module):
    def __init__(self, config: UNet15Config):
        self.conv_in = tp.Conv(4, 320, (3, 3), padding=((1, 1), (1, 1)))
        self.time_embedding = TimestepEmbedding(320, 1280)
        self.down_blocks = [
            CrossAttnDownBlock2D(320, 320, 8, 40),
            CrossAttnDownBlock2D(320, 640, 8, 80),
            CrossAttnDownBlock2D(640, 1280, 8, 160),
            DownBlock2D(1280),
        ]
        self.mid_block = UNetMidBlock2DCrossAttn(1280, 8, 160)
        self.up_blocks = [
            UpBlock2D(2560, 1280),
            CrossAttnUpBlock2D([2560, 2560, 1920], 1280, 8, 160),
            CrossAttnUpBlock2D([1920, 1280, 960], 640, 8, 80),
            CrossAttnUpBlock2D([960, 640, 640], 320, 8, 40, use_upsampler=False),
        ]
        self.conv_norm_out = tp.GroupNorm(32, 320)
        self.conv_act = tp.silu
        self.conv_out = tp.Conv(320, 4, (3, 3), padding=((1, 1), (1, 1)))

    def __call__(self, x, timesteps=None, context=None):
        # TODO: real time embedding
        t_emb = timestep_embedding(timesteps, 320)
        emb = self.time_embedding(t_emb)

        x = self.conv_in(x)
        saved_inputs = [x]

        for block in self.down_blocks:
            if isinstance(block, DownBlock2D):
                x, inputs = block(x, emb)
            else:
                x, inputs = block(x, emb, context)
            saved_inputs += inputs

        x = self.mid_block(x, emb, context)

        for block in self.up_blocks:
            partial_inputs = saved_inputs[-3:]
            del saved_inputs[-3:]
            if isinstance(block, UpBlock2D):
                x = block(x, emb, partial_inputs)
            else:
                x = block(x, emb, context, partial_inputs)

        act = self.conv_out(self.conv_act(self.conv_norm_out(x)))
        return act

