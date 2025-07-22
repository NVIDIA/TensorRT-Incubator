# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch
import nvtripy as tp

from diffusers import StableDiffusionPipeline


def load_weights_from_hf(model, hf_model, dtype):
    tripy_state_dict = model.state_dict()

    hf_state_dict = hf_model.state_dict()
    hf_keys = hf_state_dict.keys()

    torch_dtype = getattr(torch, dtype.name)
    for key in hf_keys:
        weight = hf_state_dict[key]
        weight = weight.to(torch_dtype)
        param = tp.Tensor(weight.contiguous())
        tripy_state_dict[key.removeprefix("text_model.")] = param

    model.load_state_dict(tripy_state_dict)


def load_from_diffusers(model, dtype, hf_token):
    model_id = "KiwiXR/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=hf_token)

    load_weights_from_hf(model.text_encoder, pipe.text_encoder, dtype)
    load_weights_from_hf(model.backbone, pipe.unet, dtype)
    load_weights_from_hf(model.vae, pipe.vae, dtype)
