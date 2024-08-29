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
    resolution: int = 256
    channel_mult: Tuple[int] = (1, 2, 4, 4)


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

        b, c, h, w = h_.shape
        h_flat = tp.transpose(tp.reshape(h_, (b, c, h * w)), 1, 2)
        q, k, v = self.to_q(h_flat), self.to_k(h_flat), self.to_v(h_flat)

        # compute attention
        h_ = scaled_dot_product_attention(q, k, v, embedding_dim=self.in_channels)
        out = tp.reshape(
            tp.transpose(self.to_out[0](h_), 1, 2),
            (b, c, h, w),
        )
        return x + out

# Not to be confused with ResBlock. Called ResnetBlock2D in HF diffusers
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
        bs, c, py, px = x.shape
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
    def __init__(self, config: VAEConfig):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = tp.Conv(8, 8, (1, 1))
        self.post_quant_conv = tp.Conv(4, 4, (1, 1))

    def __call__(self, x):
        latent = self.encoder(x)
        latent = self.quant_conv(latent)
        latent = latent[:, 0:4]  # only the means
        latent = self.post_quant_conv(latent)
        return self.decoder(latent)