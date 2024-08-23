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

# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md
# Adapted from https://github.com/tinygrad/tinygrad/blob/master/examples/stable_diffusion.py

import gzip, math, re, os, pathlib, platform, tempfile, hashlib
import urllib.request
from functools import lru_cache, reduce
from tqdm import tqdm
from collections import namedtuple
from typing import List, Tuple, Callable, Optional, Union

import numpy as np
import tripy as tp
from dataclasses import dataclass

@dataclass
class CLIPConfig:
    vocab_size: int = 49408
    embedding_size: int = 768
    num_heads: int = 12
    max_seq_len: int = 77
    num_hidden_layers: int = 12
    dtype: "tripy.datatype" = tp.float32

@dataclass
class StableDiffusion15UNetConfig:
    io_channels: int = 4
    model_channels: int = 320
    channel_mult: Tuple[int] = (1, 2, 4, 4)
    attention_resolutions: Tuple[int] = (4, 2, 1)
    num_heads: int = 8
    context_dim: int = 768
    dtype: "tripy.datatype" = tp.float32

@dataclass
class StableDiffusionVAEConfig:
    io_channels: int = 3
    latent_channels: int = 4
    model_channel: int = 128
    resolution: int = 256
    channel_mult: Tuple[int] = (1, 2, 4, 4)
    dtype: "tripy.datatype" = tp.float32

# convenience methods adapted from tinygrad/tensor.py (https://docs.tinygrad.org/tensor/ops/)
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
        attn_mask = tp.where((attn_mask == 0), tp.ones_like(attn_mask) * -float("inf"), tp.zeros_like(attn_mask))
    qk = query @ tp.transpose(key, -2, -1) / math.sqrt(embedding_dim)
    return tp.cast(tp.softmax((qk + attn_mask) if attn_mask is not None else qk, -1), query.dtype) @ value


def sequential(input: tp.Tensor, ll: List[Callable[[tp.Tensor], tp.Tensor]]):
    """
    Applies a sequence of functions to `self` chaining the output of each function to the input of the next.
    """
    return reduce(lambda x, f: f(x), ll, input)


# TODO: change to linear layers?
class AttnBlock(tp.Module):
    def __init__(self, in_channels):
        self.group_norm = tp.GroupNorm(32, in_channels)
        self.to_q = tp.Linear(in_channels, in_channels)
        self.to_k = tp.Linear(in_channels, in_channels)
        self.to_v = tp.Linear(in_channels, in_channels)
        self.to_out = [tp.Linear(in_channels, in_channels)]
        self.in_channels = in_channels

    # adapted from AttnBlock in ldm repo
    def __call__(self, x):
        h_ = self.group_norm(x)

        b, c, h, w = h_.shape[0], h_.shape[1], h_.shape[2], h_.shape[3]
        h_flat = tp.transpose(tp.reshape(h_, (b, c, h * w)), 1, 2)
        q, k, v = self.to_q(h_flat), self.to_k(h_flat), self.to_v(h_flat)

        # compute attention
        h_ = scaled_dot_product_attention(q, k, v, embedding_dim=self.in_channels)
        out = tp.reshape(
            tp.transpose(self.to_out[0](h_), 1, 2),
            (b, c, h, w),
        )
        return x + out

# Used for the VAE, called ResnetBlock2D in HF diffusers
class ResnetBlock(tp.Module):
    def __init__(self, in_channels, out_channels=None):
        self.norm1 = tp.GroupNorm(32, in_channels)
        self.conv1 = tp.Conv(in_channels, out_channels, (3, 3), padding=((1, 1), (1, 1)))
        self.norm2 = tp.GroupNorm(32, out_channels)
        self.conv2 = tp.Conv(out_channels, out_channels, (3, 3), padding=((1, 1), (1, 1)))
        self.nonlinearity = tp.silu
        self.conv_shortcut = tp.Conv(in_channels, out_channels, (1, 1)) if in_channels != out_channels else lambda x: x

    def __call__(self, x):
        h = self.conv1(self.nonlinearity(self.norm1(x)))
        h = self.conv2(self.nonlinearity(self.norm2(h)))
        return self.conv_shortcut(x) + h
    
class Downsample(tp.Module):
    def __init__(self, channels):
        self.conv = tp.Conv(channels, channels, (3, 3), stride=(2, 2), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        return self.conv(x)


class Upsample(tp.Module):
    def __init__(self, channels):
        self.conv = tp.Conv(channels, channels, (3, 3), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        bs, c, py, px = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = tp.reshape(tp.expand(tp.reshape(x, (bs, c, py, 1, px, 1)), (bs, c, py, 2, px, 2)), (bs, c, py * 2, px * 2)) 
        return self.conv(x)

class UpDecoderBlock2D(tp.Module):
    def __init__(self, start_channels, channels, use_upsampler=True):
        self.resnets = [ResnetBlock(start_channels, channels), ResnetBlock(channels, channels), ResnetBlock(channels, channels)]
        if use_upsampler:
            self.upsamplers = [Upsample(channels)]

    def __call__(self, x):
        for resnet in self.resnets: 
            x = resnet(x)
        if hasattr(self, "upsamplers"):
            x = self.upsamplers[0](x)
        return x

class Mid(tp.Module):
    def __init__(self, block_in):
        self.attentions = [AttnBlock(block_in)]
        self.resnets = [ResnetBlock(block_in, block_in), ResnetBlock(block_in, block_in)]

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class Decoder(tp.Module):
    def __init__(self):
        self.conv_in = tp.Conv(4, 512, (3, 3), padding=((1, 1), (1, 1)))
        self.up_blocks = [UpDecoderBlock2D(512, 512), UpDecoderBlock2D(512, 512), UpDecoderBlock2D(512, 256), UpDecoderBlock2D(256, 128, use_upsampler=False)]
        self.mid_block = Mid(512)
        self.conv_norm_out = tp.GroupNorm(32, 128)
        self.conv_act = tp.silu
        self.conv_out = tp.Conv(128, 3, (3, 3), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))

class DownEncoderBlock2D(tp.Module):
    def __init__(self, start_channels, channels, use_downsampler=True):
        self.resnets = [ResnetBlock(start_channels, channels), ResnetBlock(channels, channels)]
        if use_downsampler:
            self.downsamplers = [Downsample(channels)]

    def __call__(self, x):
        for i in range(len(self.resnets)):
            x = self.resnets[i](x)
        if hasattr(self, "downsamplers"):
            x = self.downsamplers[0](x)
        return x

class Encoder(tp.Module):
    def __init__(self):
        self.conv_in = tp.Conv(3, 128, (3, 3), padding=((1, 1), (1, 1)))
        self.down_blocks = [DownEncoderBlock2D(128, 128), DownEncoderBlock2D(128, 256), DownEncoderBlock2D(256, 512), DownEncoderBlock2D(512, 512, use_downsampler=False)]
        self.mid_block = Mid(512)
        self.conv_norm_out = tp.GroupNorm(32, 512)
        self.conv_act = tp.silu
        self.conv_out = tp.Conv(512, 8, (3, 3), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        x = self.conv_in(x)
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
        x = self.mid_block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


class AutoencoderKL(tp.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = tp.Conv(8, 8, (1, 1))
        self.post_quant_conv = tp.Conv(4, 4, (1, 1))

    def __call__(self, x):
        latent = self.encoder(x)
        latent = self.quant_conv(latent)
        latent = latent[:, 0:4]  # only the means
        # print("latent", latent.shape)
        latent = self.post_quant_conv(latent)
        return self.decoder(latent)


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
        one_shape = tp.Shape(tp.ones((1,), dtype=tp.int32))
        target_shape = tp.concatenate([emb_out.shape, one_shape, one_shape], dim=0)
        # target_shape = emb_out.shape + (1, 1)
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
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
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
    def __init__(self):
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


class CLIPMLP(tp.Module):
    def __init__(self):
        self.fc1 = tp.Linear(768, 3072)
        self.fc2 = tp.Linear(3072, 768)

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = tp.sigmoid(1.702 * hidden_states) * hidden_states  # quick GELU
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPAttention(tp.Module):
    def __init__(self):
        self.embed_dim = 768
        self.num_heads = 12
        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = tp.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = tp.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = tp.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = tp.Linear(self.embed_dim, self.embed_dim)

    def __call__(self, hidden_states, causal_attention_mask):
        bsz, tgt_len, embed_dim = hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q, k, v = [
            tp.transpose(
                tp.reshape(x, (bsz, tgt_len, self.num_heads, self.head_dim)),
                1,
                2,
            )
            for x in (q, k, v)
        ]
        attn_output = scaled_dot_product_attention(
            q, k, v, embedding_dim=self.head_dim, attn_mask=causal_attention_mask
        )
        out = self.out_proj(tp.reshape(tp.transpose(attn_output, 1, 2), (bsz, tgt_len, embed_dim)))
        return out


class CLIPEncoderLayer(tp.Module):
    def __init__(self):
        self.self_attn = CLIPAttention()
        self.layer_norm1 = tp.LayerNorm(768)
        self.mlp = CLIPMLP()
        self.layer_norm2 = tp.LayerNorm(768)

    def __call__(self, hidden_states, causal_attention_mask):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(tp.Module):
    def __init__(self):
        self.layers = [CLIPEncoderLayer() for i in range(12)]

    def __call__(self, hidden_states, causal_attention_mask):
        for l in self.layers:
            hidden_states = l(hidden_states, causal_attention_mask)
        return hidden_states


class CLIPTextEmbeddings(tp.Module):
    def __init__(self):
        self.token_embedding = tp.Embedding(49408, 768)
        self.position_embedding = tp.Embedding(77, 768)

    def __call__(self, input_ids, position_ids):
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class CLIPTextTransformer(tp.Module):
    def __init__(self):
        self.embeddings = CLIPTextEmbeddings()
        self.encoder = CLIPEncoder()
        self.final_layer_norm = tp.LayerNorm(768)

    def __call__(self, input_ids):
        max_seq_length = 77 # input_ids.shape[1] 
        x = self.embeddings(input_ids, tp.reshape(tp.iota((max_seq_length,), dtype=tp.int32), (1, -1)))
        x = self.encoder(x, tp.triu(tp.full((1, 1, 77, 77), float("-inf")), 1))
        return self.final_layer_norm(x)


# equivalent to LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
    betas = np.linspace(beta_start**0.5, beta_end**0.5, n_training_steps, dtype=np.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return tp.Tensor(alphas_cumprod)


def clamp(tensor: tp.Tensor, min: int, max: int):
    return tp.minimum(tp.maximum(tensor, tp.ones_like(tensor) * min), tp.ones_like(tensor) * max)


class StableDiffusion(tp.Module):
    def __init__(self):
        self.alphas_cumprod = get_alphas_cumprod()
        self.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model=UNetModel())
        self.first_stage_model = AutoencoderKL()
        self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(
            transformer=namedtuple("Transformer", ["text_model"])(text_model=CLIPTextTransformer())
        )

    def get_x_prev_and_pred_x0(self, x, e_t, a_t, a_prev):
        temperature = 1
        sigma_t = 0
        sqrt_one_minus_at = tp.sqrt(1 - a_t)

        pred_x0 = (x - sqrt_one_minus_at * e_t) * tp.rsqrt(a_t)

        # direction pointing to x_t
        dir_xt = tp.sqrt(1.0 - a_prev - sigma_t**2) * e_t

        x_prev = tp.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0

    def get_model_output(self, unconditional_context, context, latent, timestep, unconditional_guidance_scale):
        latents = self.model.diffusion_model(
            tp.expand(latent, (2, latent.shape[1], latent.shape[2], latent.shape[3])),
            # tp.concatenate([latent, latent], dim=0), # WAR, in this case equivalent to expand
            timestep,
            tp.concatenate([unconditional_context, context], dim=0),
        )
        unconditional_latent, latent = latents[0:1], latents[1:2]
        e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
        return e_t

    def decode(self, x):
        x = self.first_stage_model.post_quant_conv(1 / 0.18215 * x)
        x = self.first_stage_model.decoder(x)

        # make image correct size and scale
        x = (x + 1.0) / 2.0
        # assumes non-batched input
        x = clamp(tp.permute(tp.reshape(x, (3, 512, 512)), (1, 2, 0)), 0, 1) * 255
        return x

    def __call__(self, unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance):
        e_t = self.get_model_output(unconditional_context, context, latent, timestep, guidance)
        x_prev, _ = self.get_x_prev_and_pred_x0(latent, e_t, alphas, alphas_prev)
        return x_prev  

