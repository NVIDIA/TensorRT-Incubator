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

from typing import Tuple

import tripy as tp
from dataclasses import dataclass

from examples.diffusion.helper import scaled_dot_product_attention

@dataclass
class VAEConfig:
    io_channels: int = 3
    latent_channels: int = 4
    model_channel: int = 128
    channel_mult_encode: Tuple[int] = (1, 1, 2, 4, 4)
    channel_mult_decode: Tuple[int] = (4, 4, 4, 2, 1)
    dtype: tp.dtype = tp.float32


class AttnBlock(tp.Module):
    def __init__(self, config: VAEConfig, in_channels):
        self.group_norm = tp.GroupNorm(32, in_channels, dtype=tp.float32)
        self.to_q = tp.Linear(in_channels, in_channels, dtype=config.dtype)
        self.to_k = tp.Linear(in_channels, in_channels, dtype=config.dtype)
        self.to_v = tp.Linear(in_channels, in_channels, dtype=config.dtype)
        self.to_out = [tp.Linear(in_channels, in_channels, dtype=config.dtype)]
        self.in_channels = in_channels
        self.dtype = config.dtype

    # adapted from AttnBlock in ldm repo
    def __call__(self, x):
        h_ = tp.cast(self.group_norm(tp.cast(x, self.group_norm.dtype)), x.dtype)

        b, c, h, w = h_.shape
        h_flat = tp.transpose(tp.reshape(h_, (b, c, h * w)), 1, 2)
        q, k, v = self.to_q(h_flat), self.to_k(h_flat), self.to_v(h_flat)

        # compute attention
        h_ = scaled_dot_product_attention(q, k, v, embedding_dim=self.in_channels, dtype=self.dtype)
        out = tp.reshape(
            tp.transpose(self.to_out[0](h_), 1, 2),
            (b, c, h, w),
        )
        return x + out

# Not to be confused with ResBlock. Called ResnetBlock2D in HF diffusers
class ResnetBlock(tp.Module):
    def __init__(self, config: VAEConfig, in_channels, out_channels=None):
        self.norm1 = tp.GroupNorm(32, in_channels, dtype=tp.float32)
        self.conv1 = tp.Conv(in_channels, out_channels, (3, 3), padding=((1, 1), (1, 1)), dtype=config.dtype)
        self.norm2 = tp.GroupNorm(32, out_channels, dtype=tp.float32)
        self.conv2 = tp.Conv(out_channels, out_channels, (3, 3), padding=((1, 1), (1, 1)), dtype=config.dtype)
        self.nonlinearity = tp.silu
        self.conv_shortcut = tp.Conv(in_channels, out_channels, (1, 1), dtype=config.dtype) if in_channels != out_channels else lambda x: x

    def __call__(self, x):
        h = self.conv1(self.nonlinearity(tp.cast(self.norm1(tp.cast(x, self.norm1.dtype)), x.dtype)))
        h = self.conv2(self.nonlinearity(tp.cast(self.norm2(tp.cast(h, self.norm2.dtype)), h.dtype)))
        return self.conv_shortcut(x) + h
    
class Downsample(tp.Module):
    def __init__(self, config, channels):
        self.conv = tp.Conv(channels, channels, (3, 3), stride=(2, 2), padding=((1, 1), (1, 1)), dtype=config.dtype)

    def __call__(self, x):
        return self.conv(x)


class Upsample(tp.Module):
    def __init__(self, config, channels):
        self.conv = tp.Conv(channels, channels, (3, 3), padding=((1, 1), (1, 1)), dtype=config.dtype)

    def __call__(self, x):
        bs, c, py, px = x.shape
        x = tp.reshape(tp.expand(tp.reshape(x, (bs, c, py, 1, px, 1)), (bs, c, py, 2, px, 2)), (bs, c, py * 2, px * 2)) 
        return self.conv(x)

class UpDecoderBlock2D(tp.Module):
    def __init__(self, config: VAEConfig, start_channels, channels, use_upsampler=True):
        self.resnets = [ResnetBlock(config, start_channels, channels), ResnetBlock(config, channels, channels), ResnetBlock(config, channels, channels)]
        if use_upsampler:
            self.upsamplers = [Upsample(config, channels)]

    def __call__(self, x):
        for resnet in self.resnets: 
            x = resnet(x)
        if hasattr(self, "upsamplers"):
            x = self.upsamplers[0](x)
        return x

class Mid(tp.Module):
    def __init__(self, config: VAEConfig, block_in):
        self.attentions = [AttnBlock(config, block_in)]
        self.resnets = [ResnetBlock(config, block_in, block_in), ResnetBlock(config, block_in, block_in)]

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class Decoder(tp.Module):
    def __init__(self, config: VAEConfig):
        up_channels = [config.model_channel * mult for mult in config.channel_mult_decode]
        num_resolutions = len(up_channels) - 1
        upsamplers = [True] * (num_resolutions - 1) + [False]
    
        self.conv_in = tp.Conv(config.latent_channels, config.model_channel * config.channel_mult_decode[0], (3, 3), padding=((1, 1), (1, 1)), dtype=config.dtype)
        self.up_blocks = [UpDecoderBlock2D(config, up_channels[i], up_channels[i+1], use_upsampler=upsamplers[i]) for i in range(num_resolutions)]
        self.mid_block = Mid(config, up_channels[0])
        self.conv_norm_out = tp.GroupNorm(32, config.model_channel, dtype=tp.float32)
        self.conv_act = tp.silu
        self.conv_out = tp.Conv(config.model_channel, config.io_channels, (3, 3), padding=((1, 1), (1, 1)), dtype=config.dtype)

    def __call__(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)
        return self.conv_out(self.conv_act(tp.cast(self.conv_norm_out(tp.cast(x, self.conv_norm_out.dtype)), x.dtype)))

class DownEncoderBlock2D(tp.Module):
    def __init__(self, config: VAEConfig, start_channels, channels, use_downsampler=True):
        self.resnets = [ResnetBlock(config, start_channels, channels), ResnetBlock(config, channels, channels)]
        if use_downsampler:
            self.downsamplers = [Downsample(config, channels)]

    def __call__(self, x):
        for i in range(len(self.resnets)):
            x = self.resnets[i](x)
        if hasattr(self, "downsamplers"):
            x = self.downsamplers[0](x)
        return x

class Encoder(tp.Module):
    def __init__(self, config: VAEConfig):
        down_channels = [config.model_channel * mult for mult in config.channel_mult_encode]
        num_resolutions = len(down_channels) - 1
        downsamplers = [True] * (num_resolutions - 1) + [False]
    
        self.conv_in = tp.Conv(config.io_channels, config.model_channel, (3, 3), padding=((1, 1), (1, 1)), dtype=config.dtype)
        self.down_blocks = [DownEncoderBlock2D(config, down_channels[i], down_channels[i+1], use_downsampler=downsamplers[i]) for i in range(num_resolutions)]
        self.mid_block = Mid(config, down_channels[-1])
        self.conv_norm_out = tp.GroupNorm(32, down_channels[-1], dtype=tp.float32)
        self.conv_act = tp.silu
        self.conv_out = tp.Conv(down_channels[-1], 8, (3, 3), padding=((1, 1), (1, 1)), dtype=config.dtype)

    def __call__(self, x):
        x = self.conv_in(x)
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
        x = self.mid_block(x)
        return self.conv_out(self.conv_act(tp.cast(self.conv_norm_out(tp.cast(x, self.conv_norm_out.dtype)), x.dtype)))


class AutoencoderKL(tp.Module):
    def __init__(self, config: VAEConfig):
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = tp.Conv(8, 8, (1, 1), dtype=config.dtype)
        self.post_quant_conv = tp.Conv(4, 4, (1, 1), dtype=config.dtype)

    def __call__(self, x):
        latent = self.encoder(x)
        latent = self.quant_conv(latent)
        latent = latent[:, 0:4]  # only the means
        latent = self.post_quant_conv(latent)
        return self.decoder(latent)