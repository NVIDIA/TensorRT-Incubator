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
    saved_engines_path = os.path.join(current_dir, "saved_engines", "image_pipeline")

    # Create the saved_engines directory if it doesn't exist
    if not os.path.exists(saved_engines_path):
        os.makedirs(saved_engines_path)

    if use_tripy_image_encoder:
        print("Start compiling image encoder...")
        start = time.time()
        tp_image_encoder = model.image_encoder

        # Use the saved_engines directory path instead of the fixed path
        executable_file = os.path.join(saved_engines_path, "image_encoder")

        if os.path.exists(executable_file):
            compiled_tp_image_encoder = tp.Executable.load(executable_file)
        else:
            compiled_tp_image_encoder = tp.compile(
                tp_image_encoder.forward,
                args=[
                    tp.InputInfo((1, 3, 1024, 1024), dtype=tp.float32),
                ],
            )
            print(f"Compile image encoder took {time.time() - start}s")
            compiled_tp_image_encoder.save(executable_file)

        model.image_encoder = compiled_tp_image_encoder

    if cfg["model"].use_tripy_mask_decoder:
        tp_sam_mask_decoder = model.sam_mask_decoder

        # Use the saved_engines directory path instead of the fixed path
        executable_file = os.path.join(saved_engines_path, "compiled_tp_sam_mask_decoder")
        if os.path.exists(executable_file):
            compiled_tp_sam_mask_decoder = tp.Executable.load(executable_file)
        else:
            start = time.time()
            compiled_tp_sam_mask_decoder = tp.compile(
                tp_sam_mask_decoder,
                args=[
                    tp.InputInfo((1, 256, 64, 64), dtype=tp.float32),  # image_embeddings
                    tp.InputInfo((1, 256, 64, 64), dtype=tp.float32),  # image_pe
                    tp.InputInfo((1, 2, 256), dtype=tp.float32),  # sparse_prompt_embeddings
                    tp.InputInfo((1, 256, 64, 64), dtype=tp.float32),  # dense_prompt_embeddings
                    True,  # multimask_output
                    False,  # repeat_image
                    tp.InputInfo((1, 32, 256, 256), dtype=tp.float32),  # high_res_features_1
                    tp.InputInfo((1, 64, 128, 128), dtype=tp.float32),  # high_res_features_2
                ],
            )
            print(f"Compile mask decoder took {time.time() - start}s")
            compiled_tp_sam_mask_decoder.save(executable_file)

        assert os.path.exists(executable_file)

        conv_s0 = tp.compile(model.sam_mask_decoder.conv_s0, args=[tp.InputInfo((1, 256, 256, 256), dtype=tp.float32)])
        conv_s1 = tp.compile(model.sam_mask_decoder.conv_s1, args=[tp.InputInfo((1, 256, 128, 128), dtype=tp.float32)])

        model.sam_mask_decoder = compiled_tp_sam_mask_decoder
        setattr(model.sam_mask_decoder, "conv_s0", conv_s0)
        setattr(model.sam_mask_decoder, "conv_s1", conv_s1)

    if cfg["model"].use_tripy_prompt_encoder:
        start = time.time()
        tp_prompt_encoder = model.sam_prompt_encoder
        executable_file = os.path.join(saved_engines_path, "sam_prompt_encoder")
        if os.path.exists(executable_file):
            compiled_prompt_encoder = tp.Executable.load(executable_file)
        else:
            compiled_prompt_encoder = tp.compile(
                tp_prompt_encoder,
                args=[
                    tp.InputInfo((1, 1, 2), dtype=tp.float32),
                    tp.InputInfo((1, 1), dtype=tp.int32),
                    None,
                    None,
                ],
            )
            compiled_prompt_encoder.save(executable_file)
            print(f"Compile prompt encoder took {time.time() - start}s")

        start = time.time()
        tp_dense_pe = model.sam_prompt_encoder.get_dense_pe
        compiled_get_dense_pe = tp.compile(tp_dense_pe)
        print(f"Compile dense pe took {time.time() - start}s")
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
    _load_checkpoint(model, ckpt_path, cfg)

    if cfg["model"].use_tripy_mask_decoder:
        model_dtype = getattr(tp, cfg["model"].tripy_mask_decoder_dtype)

        tp_sam_mask_decoder = model.sam_mask_decoder
        model_name_false = f"compiled_tp_sam_mask_decoder_multimask_output_false_{model_dtype}.json"
        model_name_true = f"compiled_tp_sam_mask_decoder_multimask_output_true_{model_dtype}.json"

        # executable_file = os.path.join("/tripy/examples/sam2_upstream/segment-anything-2/saved_engines/", model_name_false)

        current_dir = os.getcwd()
        saved_engines_path = os.path.join(current_dir, "saved_engines", "video_pipeline")

        # Create the saved_engines directory if it doesn't exist
        if not os.path.exists(saved_engines_path):
            os.makedirs(saved_engines_path)

        # Use the saved_engines directory path instead of the fixed path
        executable_file = os.path.join(saved_engines_path, model_name_false)

        if os.path.exists(executable_file):
            compiled_tp_sam_mask_decoder_multimask_output_false = tp.Executable.load(executable_file)
            executable_file = os.path.join(saved_engines_path, model_name_true)
            compiled_tp_sam_mask_decoder_multimask_output_true = tp.Executable.load(executable_file)
        else:
            start = time.time()
            compiled_tp_sam_mask_decoder_multimask_output_false = tp.compile(
                tp_sam_mask_decoder,
                args=[
                    tp.InputInfo((1, 256, 64, 64), dtype=model_dtype),  # image_embeddings
                    tp.InputInfo((1, 256, 64, 64), dtype=model_dtype),  # image_pe
                    tp.InputInfo((1, 3, 256), dtype=model_dtype),  # sparse_prompt_embeddings
                    tp.InputInfo((1, 256, 64, 64), dtype=model_dtype),  # dense_prompt_embeddings
                    False,  # multimask_output
                    False,  # repeat_image
                    tp.InputInfo((1, 32, 256, 256), dtype=model_dtype),  # high_res_features_1
                    tp.InputInfo((1, 64, 128, 128), dtype=model_dtype),  # high_res_features_2
                ],
            )
            print(f"Compile took {time.time() - start}s")
            compiled_tp_sam_mask_decoder_multimask_output_false.save(executable_file)

            compiled_tp_sam_mask_decoder_multimask_output_true = tp.compile(
                tp_sam_mask_decoder,
                args=[
                    tp.InputInfo((1, 256, 64, 64), dtype=model_dtype),  # image_embeddings
                    tp.InputInfo((1, 256, 64, 64), dtype=model_dtype),  # image_pe
                    tp.InputInfo((1, 2, 256), dtype=model_dtype),  # sparse_prompt_embeddings
                    tp.InputInfo((1, 256, 64, 64), dtype=model_dtype),  # dense_prompt_embeddings
                    True,  # multimask_output
                    False,  # repeat_image
                    tp.InputInfo((1, 32, 256, 256), dtype=model_dtype),  # high_res_features_1
                    tp.InputInfo((1, 64, 128, 128), dtype=model_dtype),  # high_res_features_2
                ],
            )
            print(f"Compile took {time.time() - start}s")
            executable_file = os.path.join(saved_engines_path, model_name_true)
            compiled_tp_sam_mask_decoder_multimask_output_true.save(executable_file)

        model.sam_mask_decoder.conv_s0 = tp.compile(
            model.sam_mask_decoder.conv_s0, args=[tp.InputInfo((1, 256, 256, 256), dtype=model_dtype)]
        )
        model.sam_mask_decoder.conv_s1 = tp.compile(
            model.sam_mask_decoder.conv_s1, args=[tp.InputInfo((1, 256, 128, 128), dtype=model_dtype)]
        )

        model.sam_mask_decoder_false = compiled_tp_sam_mask_decoder_multimask_output_false
        model.sam_mask_decoder_true = compiled_tp_sam_mask_decoder_multimask_output_true

    if cfg["model"].use_tripy_prompt_encoder:
        start = time.time()
        tp_prompt_encoder = model.sam_prompt_encoder
        mask_input_size = model.sam_prompt_encoder.mask_input_size
        compiled_prompt_encoder = tp.compile(
            tp_prompt_encoder,
            args=[
                tp.InputInfo((1, (1, 2, 4), 2), dtype=tp.float32),
                tp.InputInfo((1, (1, 2, 4)), dtype=tp.int32),
                None,
                None,
            ],
        )
        print(f"Compile took {time.time() - start}s")

        start = time.time()
        tp_dense_pe = model.sam_prompt_encoder.get_dense_pe
        compiled_get_dense_pe = tp.compile(tp_dense_pe)
        print(f"Compile took {time.time() - start}s")
        model.sam_prompt_encoder = compiled_prompt_encoder
        setattr(model.sam_prompt_encoder, "get_dense_pe", compiled_get_dense_pe)
        setattr(model.sam_prompt_encoder, "mask_input_size", mask_input_size)

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
        nb_keys = 0
        nb_prompt_keys = 0
        model_dtype = cfg["model"].tripy_mask_decoder_dtype
        torch_dtype = getattr(torch, model_dtype)
        for key in sd:
            if use_tripy_mask_decoder and key.startswith("sam_mask_decoder"):
                new_key = key.replace("sam_mask_decoder.", "")
                nb_keys += 1
                weight = sd[key]
                if not any(substring in key for substring in ["ln", "norm", "output_upscaling.1"]):
                    weight = weight.to(torch_dtype)
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
            tp_mask_decoder.load_state_dict(tp_mask_decoder_state_dict)

        if use_tripy_prompt_encoder:
            print(f"expected keys {len(tp_prompt_encoder_state_dict.keys())}, got {nb_prompt_keys}")
            tp_prompt_encoder.load_state_dict(tp_prompt_encoder_state_dict)

        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        # if unexpected_keys:
        #     logging.error(unexpected_keys)
        #     raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
