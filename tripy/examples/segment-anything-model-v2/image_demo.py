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

import argparse
import cv2
import os
import time
import numpy as np
import torch
import nvtripy as tp
import matplotlib.pyplot as plt

plt.switch_backend("agg")  # Switch to non-interactive backend
from PIL import Image
from typing import Optional, Dict

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from demo_utils import process_and_show_mask, show_box, show_points

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch", type=int, default=2, help="batch size of the input images, between [1, 4]")
parser.add_argument("-t", "--type", type=str, default="large", choices=["large", "small", "tiny"], help="type of the sam2 model")


def process_predictions(
    image: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    logits: np.ndarray,
    point_coords: Optional[np.ndarray] = None,
    box_coords: Optional[np.ndarray] = None,
    input_labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Process and visualize predictions, returning a dictionary containing processed masks, scores, and logits.
    """
    processed_masks = []

    # Create output directory if it doesn't exist
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    for i, (mask, score) in enumerate(zip(masks, scores)):

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)

        processed_mask = process_and_show_mask(mask, ax)
        processed_masks.append(processed_mask)

        if point_coords is not None:
            assert input_labels is not None, "Input labels required for point prompts"
            show_points(point_coords, input_labels, ax)

        if box_coords is not None:
            show_box(box_coords, ax)

        if len(scores) > 1:
            ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)

        ax.axis("off")

        if save_path:
            plt.savefig(os.path.join(save_path, f"mask_{i}_score_{score:.3f}.png"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"Scores for each prediction: {' '.join(map(str, scores))}")

    return {
        "masks": np.array(processed_masks),
        "scores": scores,
        "logits": logits,
    }


def main(image_path: str, save_path: Optional[str] = None):
    """
    Main execution function.

    Args:
        image_path (str): Path to input image
        save_path (str, optional): Directory to save visualizations

    Returns:
        Dict[str, np.ndarray]: Processing results
    """

    args = parser.parse_args()

    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))
    image_list = [image] * args.batch

    # Initialize SAM2 model
    sam2_type = args.type
    sam2_checkpoint = f"./checkpoints/sam2.1_hiera_{sam2_type}.pt"
    model_cfg = f"sam2_hiera_{sam2_type[0]}.yaml"
    device = torch.device("cuda")
    sam2_model = build_sam2(
        model_cfg,
        sam2_checkpoint,
        device=device,
    )

    # Create predictor and process image
    predictor = SAM2ImagePredictor(sam2_model)

    # Set input prompt
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    def time_function(func, num_warmup=5, num_runs=100, description=""):
        # Warmup runs
        for _ in range(num_warmup):
            func()
            tp.default_stream().synchronize()
            torch.cuda.synchronize()

        # Actual timing
        start = time.perf_counter()
        for _ in range(num_runs):
            output = func()
            tp.default_stream().synchronize()
            torch.cuda.synchronize()

        end = time.perf_counter()

        avg_time_ms = (end - start) * 1000 / num_runs
        print(
            f"{description} took {avg_time_ms:.2f} ms per run (averaged over {num_runs} runs, with {num_warmup} warmup runs)"
        )

        return output

    def generate_embedding():
        predictor.set_image_batch(image_list)
        return None

    def predict_masks():
        return predictor.predict_batch(
            point_coords_batch=[input_point] * args.batch,
            point_labels_batch=[input_label] * args.batch,
            multimask_output=True,
        )

    predictor.reset_predictor()
    time_function(generate_embedding, description="Generating image embedding")
    masks, scores, logits = time_function(predict_masks, description="Predicting masks")

    masks = masks[0]
    scores = scores[0]
    logits = logits[0]

    # Sort masks by confidence score
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    # Process and display results
    results = process_predictions(
        image,
        masks,
        scores,
        logits,
        point_coords=input_point,
        input_labels=input_label,
        save_path=save_path,
    )
    return results


if __name__ == "__main__":
    main("truck.jpg", save_path="output")
