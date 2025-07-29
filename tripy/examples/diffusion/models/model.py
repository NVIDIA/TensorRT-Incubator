#
# SPDX-FileCopyrightText: Copyright (c) 2025-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvtripy as tp
from typing import Optional
from dataclasses import dataclass, field

from models.clip_model import CLIPTextTransformer, CLIPConfig
from models.unet_model import UNetModel, UNetConfig
from models.vae_model import AutoencoderKL, VAEConfig
from models.utils import clamp


@dataclass
class StableDiffusionConfig:
    dtype: tp.dtype
    clip_config: Optional[CLIPConfig] = field(default=None, init=False)
    unet_config: Optional[UNetConfig] = field(default=None, init=False)
    vae_config: Optional[VAEConfig] = field(default=None, init=False)

    def __post_init__(self):
        self.clip_config = CLIPConfig(dtype=self.dtype)
        self.unet_config = UNetConfig(dtype=self.dtype)
        self.vae_config = VAEConfig(dtype=self.dtype)


class StableDiffusion(tp.Module):
    def __init__(self, config: StableDiffusionConfig):
        self.text_encoder = CLIPTextTransformer(config.clip_config)
        self.backbone = UNetModel(config.unet_config)
        self.vae = AutoencoderKL(config.vae_config)

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
        latents = self.backbone(
            tp.expand(latent, (2, latent.shape[1], latent.shape[2], latent.shape[3])),
            timestep,
            tp.concatenate([unconditional_context, context], dim=0),
        )
        unconditional_latent, latent = latents[0:1], latents[1:2]
        e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
        return e_t

    def decode(self, x):
        x = self.vae.post_quant_conv(1 / 0.18215 * x)
        x = self.vae.decoder(x)

        # make image correct size and scale
        x = (x + 1.0) / 2.0
        # assumes non-batched input
        x = clamp(tp.permute(tp.reshape(x, (3, 512, 512)), (1, 2, 0)), 0, 1) * 255
        return x

    def __call__(
        self, unconditional_context, context, latent, timesteps, alphas_cumprod, alphas_cumprod_prev, guidance, index
    ):
        timestep = tp.reshape(timesteps[index], (1,))
        alphas = alphas_cumprod[index]
        alphas_prev = alphas_cumprod_prev[index]
        e_t = self.get_model_output(unconditional_context, context, latent, timestep, guidance)
        x_prev, _ = self.get_x_prev_and_pred_x0(latent, e_t, alphas, alphas_prev)
        return x_prev
