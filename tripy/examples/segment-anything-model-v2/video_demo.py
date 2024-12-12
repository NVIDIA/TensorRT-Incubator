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

import torch
from sam2.build_sam import build_sam2_video_predictor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from demo_utils import process_and_show_mask as show_mask
from typing import Optional

import os
import time


def main(video_dir: str, save_path: Optional[str] = None):
    """
    Main execution function.

    Args:
        video_path (str): Path to where video frames are stored
        save_path (str, optional): Directory to save visualizations

    Returns:
        Dict[str, np.ndarray]: Processing results
    """

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cuda"))

    # scan all the JPEG frame names in this directory
    frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    # take a look the first video frame
    frame_idx = 0
    if save_path:
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
        plt.savefig(os.path.join(save_path, f"video_{frame_idx}.png"))
        plt.close("all")

    inference_state = predictor.init_state(video_path=video_dir)

    def make_tensors_contiguous(d):
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                d[key] = value.contiguous()
        return d

    inference_state = make_tensors_contiguous(inference_state)

    predictor.reset_state(inference_state)

    prompts = {}  # hold all the clicks we add for visualization

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a positive click at (x, y) = (200, 300) to get started on the first object
    points = np.array([[200, 300]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    prompts[ann_obj_id] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # add the first object
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
    # sending all clicks (and their labels) to `add_new_points_or_box`
    points = np.array([[200, 300], [275, 175]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 0], np.int32)
    prompts[ann_obj_id] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)

    # Let's now move on to the second object we want to track (giving it object id `3`)
    # with a positive click at (x, y) = (400, 150)
    points = np.array([[400, 150]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    prompts[ann_obj_id] = points, labels

    # `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # run propagation throughout the video and collect the results in a dict
    start = time.perf_counter()
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0) for i, out_obj_id in enumerate(out_obj_ids)
        }
    end = time.perf_counter()
    print(f"Video segmentation took {(end - start)}s")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        # render the segmentation results every few frames
        vis_frame_stride = 30
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask.cpu().numpy(), plt.gca(), obj_id=out_obj_id)
            plt.savefig(os.path.join(save_path, f"video_final_mask_{out_frame_idx}.png"))


if __name__ == "__main__":
    main("./bedroom", save_path="output")
