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
import torch
import tripy as tp

from diffusers import StableDiffusionPipeline


def load_weights_from_hf(model, hf_model, dtype, debug=False):
    tripy_state_dict = model.state_dict()
    tripy_keys = tripy_state_dict.keys()

    hf_state_dict = hf_model.state_dict()
    hf_keys = hf_state_dict.keys()

    assert_msg = f"Mismatched keys: {hf_keys} != {tripy_keys}"
    if debug and len(hf_keys) != len(tripy_keys):
        print("\nERROR: unused weights in HF_state_dict:\n", sorted(list(hf_keys - tripy_keys)))
        print("\nERROR: unused weights in tripy_state_dict:\n", sorted(list(tripy_keys - hf_keys)))
        assert_msg = f"Mismatched keys between HF and provided model, see above."
    assert len(hf_keys) == len(tripy_keys), assert_msg

    torch_dtype = getattr(torch, dtype.name)
    for key in hf_keys:
        weight = hf_state_dict[key]
        if "norm" not in key:
            weight = weight.to(torch_dtype)
        param = tp.Tensor(weight.contiguous())
        tripy_state_dict[key.removeprefix("text_model.")] = param

    model.load_state_dict(tripy_state_dict)


def load_from_diffusers(model, dtype, hf_token, debug=False):
    model_id = "KiwiXR/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=hf_token)

    load_weights_from_hf(model.cond_stage_model.transformer.text_model, pipe.text_encoder, dtype, debug=debug)
    load_weights_from_hf(model.model.diffusion_model, pipe.unet, dtype, debug=debug)
    load_weights_from_hf(model.first_stage_model, pipe.vae, dtype, debug=debug)
