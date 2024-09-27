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


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
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

    if cfg["model"].use_tripy_mask_decoder:
        start = time.time()
        tp_sam_mask_decoder = model.sam_mask_decoder
        compiler = tp.Compiler(tp_sam_mask_decoder)
        compiled_tp_sam_mask_decoder = compiler.compile(
            tp.InputInfo((1, 256, 64, 64), dtype=tp.float32),  # image_embeddings
            tp.InputInfo((1, 256, 64, 64), dtype=tp.float32),  # image_pe
            tp.InputInfo((1, 2, 256), dtype=tp.float32),  # sparse_prompt_embeddings
            tp.InputInfo((1, 256, 64, 64), dtype=tp.float32),  # dense_prompt_embeddings
            True,  # multimask_output
            False,  # repeat_image
            tp.InputInfo((1, 32, 256, 256), dtype=tp.float32),  # high_res_features_1
            tp.InputInfo((1, 64, 128, 128), dtype=tp.float32),  # high_res_features_2
        )
        print(f"Compile took {time.time() - start}s")
        conv_s0 = model.sam_mask_decoder.conv_s0
        conv_s1 = model.sam_mask_decoder.conv_s1

        model.sam_mask_decoder = compiled_tp_sam_mask_decoder
        setattr(model.sam_mask_decoder, "conv_s0", conv_s0)
        setattr(model.sam_mask_decoder, "conv_s1", conv_s1)

    if cfg["model"].use_tripy_prompt_encoder:
        start = time.time()
        tp_prompt_encoder = model.sam_prompt_encoder
        compiler = tp.Compiler(tp_prompt_encoder)
        compiled_prompt_encoder = compiler.compile(
            tp.InputInfo((1, 1, 2), dtype=tp.float32),
            tp.InputInfo((1, 1), dtype=tp.int32),
            None,
            None,
        )
        print(f"Compile took {time.time() - start}s")

        start = time.time()
        tp_dense_pe = model.sam_prompt_encoder.get_dense_pe
        compiler = tp.Compiler(tp_dense_pe)
        compiled_get_dense_pe = compiler.compile()
        print(f"Compile took {time.time() - start}s")
        model.sam_prompt_encoder = compiled_prompt_encoder
        setattr(model.sam_prompt_encoder, "get_dense_pe", compiled_get_dense_pe)

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
    _load_checkpoint(model, ckpt_path)
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
    return build_sam2_video_predictor(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def _load_checkpoint(model, ckpt_path, cfg=None):

    use_tripy_mask_decoder = cfg["model"].use_tripy_mask_decoder
    use_tripy_prompt_encoder = cfg["model"].use_tripy_prompt_encoder
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        tp_mask_decoder = model.sam_mask_decoder
        tp_mask_decoder_state_dict = tp_mask_decoder.state_dict()

        tp_prompt_encoder = model.sam_prompt_encoder
        tp_prompt_encoder_state_dict = tp_prompt_encoder.state_dict()

        expected_mask_decoder_keys = len(tp_mask_decoder_state_dict.keys())
        print(f"Tripy transformer state_dict has {expected_mask_decoder_keys} keys.")
        nb_keys = 0
        nb_prompt_keys = 0

        for key in sd:
            if use_tripy_mask_decoder and key.startswith("sam_mask_decoder"):
                new_key = key.replace("sam_mask_decoder.", "")
                nb_keys += 1
                weight = sd[key]
                param = tp.Parameter(weight)
                tp_mask_decoder_state_dict[new_key] = param

            if use_tripy_prompt_encoder and key.startswith("sam_prompt_encoder"):
                new_key = key.replace("sam_prompt_encoder.", "")
                nb_prompt_keys += 1
                weight = sd[key]
                param = tp.Parameter(weight)
                tp_prompt_encoder_state_dict[new_key] = param

        if use_tripy_mask_decoder:
            print(f"expected keys {expected_mask_decoder_keys}, got {nb_keys}")
            tp_mask_decoder.load_from_state_dict(tp_mask_decoder_state_dict)

        if use_tripy_prompt_encoder:
            print(f"expected keys {len(tp_prompt_encoder_state_dict.keys())}, got {nb_prompt_keys}")
            tp_prompt_encoder.load_from_state_dict(tp_prompt_encoder_state_dict)

        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        # if unexpected_keys:
        #     logging.error(unexpected_keys)
        #     raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
