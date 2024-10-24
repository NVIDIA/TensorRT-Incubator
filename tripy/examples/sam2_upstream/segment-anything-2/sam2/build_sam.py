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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import tripy as tp
import time
import tempfile
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
    return {
        "memory_attention": {
            "enabled": isinstance(model.memory_attention, tp.Module),
            "model": model.memory_attention,
            "dtype": getattr(cfg["model"].memory_attention, "dtype", "float32"),
            "compile_args": [
                tp.InputInfo(
                    (4096, 1, 256),
                    getattr(
                        tp, getattr(cfg["model"].memory_attention, "dtype", "float32")
                    ),
                ),
                tp.InputInfo(
                    ((3277, 8196, 81925), 1, 64),
                    getattr(
                        tp, getattr(cfg["model"].memory_attention, "dtype", "float32")
                    ),
                ),
                tp.InputInfo(
                    (4096, 1, 256),
                    getattr(
                        tp, getattr(cfg["model"].memory_attention, "dtype", "float32")
                    ),
                ),
                tp.InputInfo(
                    ((3277, 8196, 81925), 1, 64),
                    getattr(
                        tp, getattr(cfg["model"].memory_attention, "dtype", "float32")
                    ),
                ),
                tp.InputInfo(((1, 1641, 3282),), tp.int32),
            ],
            "skip_dtype_convert": ["ln", "norm"],
        },
        "sam_mask_decoder_false": {
            "enabled": getattr(cfg["model"], "use_tripy_mask_decoder", False),
            "model": model.sam_mask_decoder,
            "dtype": getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32"),
            "compile_args": [
                tp.InputInfo(
                    (1, 256, 64, 64),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # image_embeddings
                tp.InputInfo(
                    (1, 256, 64, 64),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # image_pe
                tp.InputInfo(
                    (1, 3, 256),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # sparse_prompt_embeddings
                tp.InputInfo(
                    (1, 256, 64, 64),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # dense_prompt_embeddings
                False,  # multimask_output
                False,  # repeat_image
                tp.InputInfo(
                    (1, 32, 256, 256),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # high_res_features_1
                tp.InputInfo(
                    (1, 64, 128, 128),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # high_res_features_2
            ],
            "skip_dtype_convert": ["ln", "norm", "output_upscaling.1"],
        },
        "sam_mask_decoder_true": {
            "enabled": getattr(cfg["model"], "use_tripy_mask_decoder", False),
            "model": model.sam_mask_decoder,
            "dtype": getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32"),
            "compile_args": [
                tp.InputInfo(
                    (1, 256, 64, 64),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # image_embeddings
                tp.InputInfo(
                    (1, 256, 64, 64),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # image_pe
                tp.InputInfo(
                    (1, 2, 256),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # sparse_prompt_embeddings
                tp.InputInfo(
                    (1, 256, 64, 64),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # dense_prompt_embeddings
                True,  # multimask_output
                False,  # repeat_image
                tp.InputInfo(
                    (1, 32, 256, 256),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # high_res_features_1
                tp.InputInfo(
                    (1, 64, 128, 128),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                ),  # high_res_features_2
            ],
            "skip_dtype_convert": ["ln", "norm", "output_upscaling.1"],
            "skip_load_state_dict": True,
        },
        "sam_mask_decoder.conv_s0": {
            "enabled": getattr(cfg["model"], "use_tripy_mask_decoder", False),
            "model": model.sam_mask_decoder.conv_s0,
            "dtype": getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32"),
            "compile_args": [
                tp.InputInfo(
                    (1, 256, 256, 256),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                )
            ],
            "skip_dtype_convert": [],
            "skip_load_state_dict": True,
        },
        "sam_mask_decoder.conv_s1": {
            "enabled": getattr(cfg["model"], "use_tripy_mask_decoder", False),
            "model": model.sam_mask_decoder.conv_s1,
            "dtype": getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32"),
            "compile_args": [
                tp.InputInfo(
                    (1, 256, 128, 128),
                    dtype=getattr(
                        tp, getattr(cfg["model"], "tripy_mask_decoder_dtype", "float32")
                    ),
                )
            ],
            "skip_dtype_convert": [],
            "skip_load_state_dict": True,
        },
        "memory_encoder": {
            "enabled": isinstance(model.memory_encoder, tp.Module),
            "model": model.memory_encoder,
            "dtype": "float32",  # TODO add fp16 to yaml
            "compile_args": [
                tp.InputInfo((1, 256, 64, 64), tp.float32),
                tp.InputInfo((1, 1, 1024, 1024), tp.float32),
                True,
            ],
            "skip_dtype_convert": [],
        },
        "sam_prompt_encoder": {
            "enabled": getattr(cfg["model"], "use_tripy_prompt_encoder", False),
            "model": model.sam_prompt_encoder,
            "dtype": "float32",  # TODO add fp16 to yaml
            "compile_args": [
                tp.InputInfo((1, (1, 2, 4), 2), dtype=tp.float32),
                tp.InputInfo((1, (1, 2, 4)), dtype=tp.int32),
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
            "enabled": getattr(cfg["model"], "use_tripy_prompt_encoder", False),
            "model": model.sam_prompt_encoder.get_dense_pe,
            "dtype": "float32",  # TODO add fp16 to yaml
            "compile_args": [],
            "skip_dtype_convert": [],
            "skip_load_state_dict": True,
        },
        "image_encoder.compiled_executable": {
            "enabled": isinstance(model.image_encoder, tp.Module),
            "model": model.image_encoder.forward,
            "dtype": getattr(cfg["model"].image_encoder.trunk, "dtype", "float32"),
            "compile_args": [
                tp.InputInfo(
                    (1, 3, 1024, 1024),
                    dtype=getattr(
                        tp,
                        getattr(cfg["model"].image_encoder.trunk, "dtype", "float32"),
                    ),
                ),
            ],
            "skip_dtype_convert": ["norm"],
            "special_key_loading": lambda key: (
                # If it's a neck.convs key that contains 'conv.'
                # neck.convs.0.conv.weight -> neck.convs.0.weight
                ".".join(parts[:-2] + [parts[-1]])
                if (parts := key.split("."))
                and key.startswith("neck.convs")
                and "conv." in key
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


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    use_tripy_image_encoder=False,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)

    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, cfg)

    current_dir = os.getcwd()
    saved_engines_path = os.path.join(current_dir, "saved_engines")

    # Create the saved_engines directory if it doesn't exist
    if not os.path.exists(saved_engines_path):
        os.makedirs(saved_engines_path)

    # Get component configurations
    components = get_component_configs(model, cfg)
    required_components_for_image = [
        "sam_mask_decoder_true",
        "sam_mask_decoder.conv_s0",
        "sam_mask_decoder.conv_s1",
        "sam_prompt_encoder",
        "sam_prompt_encoder.get_dense_pe",
        "image_encoder.compiled_executable",
    ]

    for comp_name, comp_info in components.items():
        if not comp_info["enabled"] or comp_name not in required_components_for_image:
            continue

        executable_file = os.path.join(saved_engines_path, comp_name)
        if os.path.exists(executable_file):
            print(f"Loading existing compiled {comp_name} from {executable_file}")
            compiled_model = tp.Executable.load(executable_file)
        else:
            print(f"Compiling {comp_name}...")
            start = time.time()
            compiled_model = tp.compile(
                comp_info["model"], args=comp_info["compile_args"]
            )
            print(f"Compilation took {time.time() - start:.2f}s")
            compiled_model.save(executable_file)

        old_model = comp_info["model"]
        # If model is model.forward, retrieve the original model object
        if hasattr(old_model, "__self__"):
            old_model = old_model.__self__

        set_model_attr(model, comp_name, compiled_model)
        if (
            "special_handling" in comp_info
            and comp_info["special_handling"] is not None
        ):
            comp_info["special_handling"](old_model)

    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, cfg)

    current_dir = os.getcwd()
    saved_engines_path = os.path.join(current_dir, "saved_engines")
    # Create the saved_engines directory if it doesn't exist
    if not os.path.exists(saved_engines_path):
        os.makedirs(saved_engines_path)

    # Get component configurations
    components = get_component_configs(model, cfg)

    for comp_name, comp_info in components.items():
        if not comp_info["enabled"]:
            continue

        executable_file = os.path.join(saved_engines_path, comp_name)
        if os.path.exists(executable_file):
            print(f"Loading existing compiled {comp_name} from {executable_file}")
            compiled_model = tp.Executable.load(executable_file)
        else:
            print(f"Compiling {comp_name}...")
            start = time.time()
            compiled_model = tp.compile(
                comp_info["model"], args=comp_info["compile_args"]
            )
            print(f"Compilation took {time.time() - start:.2f}s")
            compiled_model.save(executable_file)

        old_model = comp_info["model"]
        # If model is model.forward, retrieve the original model object
        if hasattr(old_model, "__self__"):
            old_model = old_model.__self__

        set_model_attr(model, comp_name, compiled_model)
        if (
            "special_handling" in comp_info
            and comp_info["special_handling"] is not None
        ):
            comp_info["special_handling"](old_model)

    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def load_component_weights(comp_name, component_info, state_dict, checkpoint_dict):
    """
    Load weights for a single component from checkpoint into state dict.
    """

    converted_keys = 0
    for key in checkpoint_dict:
        # Remove _true/_false suffixes if present
        for suffix in ["_true", "_false", ".compiled_executable"]:
            if comp_name.endswith(suffix):
                comp_name = comp_name[: -len(suffix)]
                break

        if not key.startswith(comp_name) or (
            component_info.get("skip_load_state_dict") is True
        ):
            continue

        new_key = key.replace(f"{comp_name}.", "")
        if "special_key_loading" in component_info:
            new_key = component_info["special_key_loading"](new_key)
        weight = checkpoint_dict[key]

        should_convert = not any(
            skip in key for skip in component_info["skip_dtype_convert"]
        )
        if should_convert and component_info["dtype"] is not None:
            weight = weight.to(getattr(torch, component_info["dtype"]))

        state_dict[new_key] = tp.Parameter(weight.contiguous())
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

        if not comp_info["enabled"] or comp_info.get("skip_load_state_dict") is True:
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
        comp_model.load_state_dict(component_sd)
        if comp_name == "image_encoder.compiled_executable":
            # WAR: https://github.com/NVIDIA/TensorRT-Incubator/issues/269
            comp_model.trunk.pos_embed_torch = comp_model.trunk._get_pos_embed_torch(
                (256, 256)
            )

        print(f"Converted {converted_keys} keys for {comp_name}")