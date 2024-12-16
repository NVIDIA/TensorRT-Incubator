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


import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Dict, Any, Optional

import tripy as tp
import time
import os


def set_model_attr(model, attr_path, value):
    """
    Set a model attribute, handling both nested and regular attributes.
    """
    if "." not in attr_path:
        setattr(model, attr_path, value)
    else:
        attrs = attr_path.split(".")
        obj = model
        # Navigate to the parent object
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        # Set the attribute on the parent object
        setattr(obj, attrs[-1], value)


def get_component_configs(model, cfg):
    """
    Get configurations for different components, including both compilation and weight loading info.
    """
    batchsize = (1, 2, 4)
    num_obj = (1, 2, 4)
    model_precision = getattr(cfg["model"], "model_precision", "float32")
    return {
        "memory_attention": {
            "enabled": True,
            "model": model.memory_attention,
            "dtype": model_precision,
            "compile_args": [
                tp.InputInfo(
                    (4096, (1, 2, 8), 256),
                    getattr(tp, model_precision),
                ),
                tp.InputInfo(
                    ((4100, 16400, 28736), (1, 2, 8), 64),
                    getattr(tp, model_precision),
                ),
                tp.InputInfo(
                    (4096, (1, 2, 8), 256),
                    getattr(tp, model_precision),
                ),
                tp.InputInfo(
                    ((4100, 16400, 28736), (1, 2, 8), 64),
                    getattr(tp, model_precision),
                ),
                tp.InputInfo(((4, 16, 64),), tp.int32),
            ],
            "skip_dtype_convert": ["ln", "norm"],
        },
        "sam_mask_decoder_false": {
            "enabled": True,
            "model": model.sam_mask_decoder,
            "dtype": model_precision,
            "compile_args": [
                tp.InputInfo(
                    (1, 256, 64, 64),
                    dtype=getattr(tp, model_precision),
                ),  # image_embeddings
                tp.InputInfo(
                    (1, 256, 64, 64),
                    dtype=getattr(tp, model_precision),
                ),  # image_pe
                tp.InputInfo(
                    (1, 3, 256),
                    dtype=getattr(tp, model_precision),
                ),  # sparse_prompt_embeddings
                tp.InputInfo(
                    (1, 256, 64, 64),
                    dtype=getattr(tp, model_precision),
                ),  # dense_prompt_embeddings
                False,  # multimask_output
                False,  # repeat_image
                tp.InputInfo(
                    (1, 32, 256, 256),
                    dtype=getattr(tp, model_precision),
                ),  # high_res_features_1
                tp.InputInfo(
                    (1, 64, 128, 128),
                    dtype=getattr(tp, model_precision),
                ),  # high_res_features_2
            ],
            "skip_dtype_convert": ["ln", "norm", "output_upscaling.1"],
        },
        "sam_mask_decoder_true": {
            "enabled": True,
            "model": model.sam_mask_decoder,
            "dtype": model_precision,
            "compile_args": [
                tp.InputInfo(
                    (batchsize, 256, 64, 64),
                    dtype=getattr(tp, model_precision),
                ),  # image_embeddings
                tp.InputInfo(
                    (1, 256, 64, 64),
                    dtype=getattr(tp, model_precision),
                ),  # image_pe
                tp.InputInfo(
                    (batchsize, (2, 4, 6), 256),
                    dtype=getattr(tp, model_precision),
                ),  # sparse_prompt_embeddings
                tp.InputInfo(
                    (batchsize, 256, 64, 64),
                    dtype=getattr(tp, model_precision),
                ),  # dense_prompt_embeddings
                True,  # multimask_output
                False,  # repeat_image
                tp.InputInfo(
                    (batchsize, 32, 256, 256),
                    dtype=getattr(tp, model_precision),
                ),  # high_res_features_1
                tp.InputInfo(
                    (batchsize, 64, 128, 128),
                    dtype=getattr(tp, model_precision),
                ),  # high_res_features_2
            ],
            "skip_dtype_convert": ["ln", "norm", "output_upscaling.1"],
            "skip_load_state_dict": True,
        },
        "sam_mask_decoder.conv_s0": {
            "enabled": True,
            "model": model.sam_mask_decoder.conv_s0,
            "dtype": model_precision,
            "compile_args": [
                tp.InputInfo(
                    (batchsize, 256, 256, 256),
                    dtype=getattr(tp, model_precision),
                )
            ],
            "skip_dtype_convert": [],
            "skip_load_state_dict": True,
        },
        "sam_mask_decoder.conv_s1": {
            "enabled": True,
            "model": model.sam_mask_decoder.conv_s1,
            "dtype": model_precision,
            "compile_args": [
                tp.InputInfo(
                    (batchsize, 256, 128, 128),
                    dtype=getattr(tp, model_precision),
                )
            ],
            "skip_dtype_convert": [],
            "skip_load_state_dict": True,
        },
        "memory_encoder": {
            "enabled": True,
            "model": model.memory_encoder,
            "dtype": "float32",  # TODO add fp16 to yaml
            "compile_args": [
                tp.InputInfo((batchsize, 256, 64, 64), tp.float32),
                tp.InputInfo((batchsize, num_obj, 1024, 1024), tp.float32),
                True,
            ],
            "skip_dtype_convert": ["ln", "norm"]
            + [f"encoder.{i}.{param}" for i in range(1, 40, 3) for param in ("weight", "bias")],
        },
        "sam_prompt_encoder": {
            "enabled": True,
            "model": model.sam_prompt_encoder,
            "dtype": "float32",
            "compile_args": [
                tp.InputInfo((batchsize, num_obj, 2), dtype=tp.float32),
                tp.InputInfo((batchsize, num_obj), dtype=tp.int32),
                None,
                None,
            ],
            "skip_dtype_convert": [],
            "special_handling": lambda original_model: {
                setattr(
                    model.sam_prompt_encoder,
                    "mask_input_size",
                    original_model.mask_input_size,
                )
            },
        },
        "sam_prompt_encoder.get_dense_pe": {
            "enabled": True,
            "model": model.sam_prompt_encoder.get_dense_pe,
            "dtype": model_precision,
            "compile_args": [],
            "skip_dtype_convert": [],
            "skip_load_state_dict": True,
        },
        "image_encoder.compiled_executable": {
            "enabled": True,
            "model": model.image_encoder.forward,
            "dtype": model_precision,
            "compile_args": [
                tp.InputInfo(
                    (batchsize, 3, 1024, 1024),
                    dtype=getattr(
                        tp,
                        model_precision,
                    ),
                ),
            ],
            "skip_dtype_convert": ["norm"],
            "special_key_loading": lambda key: (
                # If it's a neck.convs key that contains 'conv.'
                # neck.convs.0.conv.weight -> neck.convs.0.weight
                ".".join(parts[:-2] + [parts[-1]])
                if (parts := key.split(".")) and key.startswith("neck.convs") and "conv." in key
                else key
            ),
            "special_handling": lambda original_model: {
                setattr(
                    model.image_encoder,
                    "trunk",
                    type("Trunk", (), {"dtype": original_model.trunk.dtype})(),
                )
            },
        },
    }


class SAM2ModelCache:
    """Singleton class to manage cached compiled models for SAM2."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SAM2ModelCache, cls).__new__(cls)
            cls._instance.cached_models = {}
            cls._instance.saved_engines_path = os.path.join(os.getcwd(), "saved_engines")
            if not os.path.exists(cls._instance.saved_engines_path):
                os.makedirs(cls._instance.saved_engines_path)
        return cls._instance

    def get_or_compile_component(self, comp_name: str, comp_info: Dict[str, Any]) -> tp.Executable:
        """Get a cached compiled model or compile and cache it if not exists."""
        if not comp_info["enabled"]:
            return None

        # Check if already in memory cache
        if comp_name in self.cached_models:
            print(f"Using in-memory cached model for {comp_name}")
            return self.cached_models[comp_name]

        executable_file = os.path.join(self.saved_engines_path, comp_name)

        # Check if compiled model exists on disk
        if os.path.exists(executable_file):
            print(f"Loading existing compiled {comp_name} from {executable_file}")
            compiled_model = tp.Executable.load(executable_file)
        else:
            print(f"Compiling {comp_name}...")
            start = time.time()
            compiled_model = tp.compile(comp_info["model"], args=comp_info["compile_args"])
            print(f"Compilation took {time.time() - start:.2f}s")
            compiled_model.save(executable_file)

        # Cache the compiled model in memory
        self.cached_models[comp_name] = compiled_model
        return compiled_model


def build_sam2_base(
    config_file: str,
    ckpt_path: Optional[str] = None,
    device: str = "cuda",
    mode: str = "eval",
    hydra_overrides: list = None,
    apply_postprocessing: bool = True,
    **kwargs,
) -> Any:
    """Base function for building SAM2 models with caching support."""
    if hydra_overrides is None:
        hydra_overrides = []

    if apply_postprocessing:
        hydra_overrides = hydra_overrides.copy()
        hydra_overrides += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)

    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, cfg)

    # Get component configurations and initialize cache
    components = get_component_configs(model, cfg)
    model_cache = SAM2ModelCache()

    # Compile or load all required components
    for comp_name, comp_info in components.items():
        if not comp_info["enabled"]:
            continue

        compiled_model = model_cache.get_or_compile_component(comp_name, comp_info)
        if compiled_model is not None:
            old_model = comp_info["model"]
            if hasattr(old_model, "__self__"):
                old_model = old_model.__self__

            set_model_attr(model, comp_name, compiled_model)
            if "special_handling" in comp_info and comp_info["special_handling"] is not None:
                comp_info["special_handling"](old_model)

    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2(
    config_file: str,
    ckpt_path: Optional[str] = None,
    device: str = "cuda",
    mode: str = "eval",
    hydra_overrides_extra: list = None,
    apply_postprocessing: bool = True,
    use_tripy_image_encoder: bool = False,
    **kwargs,
) -> Any:
    """Build SAM2 model with caching support."""
    return build_sam2_base(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device=device,
        mode=mode,
        hydra_overrides=hydra_overrides_extra,
        apply_postprocessing=apply_postprocessing,
        **kwargs,
    )


def build_sam2_video_predictor(
    config_file: str,
    ckpt_path: Optional[str] = None,
    device: str = "cuda",
    mode: str = "eval",
    hydra_overrides_extra: list = None,
    apply_postprocessing: bool = True,
    **kwargs,
) -> Any:
    """Build SAM2 video predictor with caching support."""
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy() if hydra_overrides_extra else []
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]

    hydra_overrides.extend(hydra_overrides_extra or [])

    return build_sam2_base(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device=device,
        mode=mode,
        hydra_overrides=hydra_overrides,
        apply_postprocessing=apply_postprocessing,
        **kwargs,
    )


def load_component_weights(comp_name, component_info, state_dict, checkpoint_dict):
    """
    Load weights for a single component from checkpoint into state dict.
    """

    converted_keys = 0
    if component_info.get("skip_load_state_dict"):
        return converted_keys

    for key in checkpoint_dict:
        # Remove _true/_false suffixes if present
        for suffix in ["_true", "_false", ".compiled_executable"]:
            if comp_name.endswith(suffix):
                comp_name = comp_name[: -len(suffix)]
                break

        if not key.startswith(comp_name):
            continue

        new_key = key.replace(f"{comp_name}.", "")
        if "special_key_loading" in component_info:
            new_key = component_info["special_key_loading"](new_key)
        weight = checkpoint_dict[key]

        should_convert = not any(skip in key for skip in component_info["skip_dtype_convert"])
        if should_convert and component_info["dtype"] is not None:
            weight = weight.to(getattr(torch, component_info["dtype"]))

        state_dict[new_key] = tp.Tensor(weight.contiguous())
        converted_keys += 1

    return converted_keys


def _load_checkpoint(model, ckpt_path, cfg=None):

    if ckpt_path is None:
        return

    sd = torch.load(ckpt_path, map_location="cpu")["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

    # Get paths for compiled models
    current_dir = os.getcwd()
    saved_engines_path = os.path.join(current_dir, "saved_engines")

    # Get component configurations
    components = get_component_configs(model, cfg)

    # Process each component
    for comp_name, comp_info in components.items():

        if not comp_info["enabled"] or comp_info.get("skip_load_state_dict"):
            continue

        # Skip if compiled model exists
        model_path = os.path.join(saved_engines_path, comp_name)
        if os.path.exists(model_path):
            print(f"Using existing compiled model for {comp_name}")
            continue

        # If no compiled model exists, convert and load weights
        print(f"Converting weights for {comp_name}")

        comp_model = comp_info["model"]
        # If model is model.forward, retrieve the original model object
        if hasattr(comp_model, "__self__"):
            comp_model = comp_model.__self__
        component_sd = comp_model.state_dict()
        converted_keys = load_component_weights(comp_name, comp_info, component_sd, sd)
        comp_model.load_state_dict(component_sd, strict=False)
        if comp_name == "image_encoder.compiled_executable":
            comp_model.trunk.generate_static_pos_embed((256, 256))

        print(f"Converted {converted_keys} keys for {comp_name}")
