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

from collections import namedtuple

import numpy as np
import tripy as tp
from typing import Optional
from dataclasses import dataclass, field

from examples.diffusion.clip_model import CLIPTextTransformer, CLIPConfig
from examples.diffusion.unet_model import UNetModel, UNetConfig
from examples.diffusion.vae_model import AutoencoderKL, VAEConfig
from examples.diffusion.helper import clamp

@dataclass
class StableDiffusionConfig:
    dtype: tp.dtype = tp.float32
    clip_config: Optional[CLIPConfig] = field(default=None, init=False)
    unet_config: Optional[UNetConfig] = field(default=None, init=False)
    vae_config: Optional[VAEConfig] = field(default=None, init=False)

    def __post_init__(self):
        self.clip_config = CLIPConfig(dtype=self.dtype)
        self.unet_config = UNetConfig(dtype=self.dtype)
        self.vae_config = VAEConfig(dtype=self.dtype)

# equivalent to LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000, dtype=tp.float32):
    betas = np.linspace(beta_start**0.5, beta_end**0.5, n_training_steps, dtype=np.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return tp.cast(tp.Tensor(alphas_cumprod), dtype)


class StableDiffusion(tp.Module):
    def __init__(self, config: StableDiffusionConfig):
        self.alphas_cumprod = get_alphas_cumprod()
        self.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model=UNetModel(config.unet_config))
        self.first_stage_model = AutoencoderKL(config.vae_config)
        self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(
            transformer=namedtuple("Transformer", ["text_model"])(text_model=CLIPTextTransformer(config.clip_config))
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

