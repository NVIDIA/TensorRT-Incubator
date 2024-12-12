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
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")  # Switch to non-interactive backend
from typing import Tuple, Optional


def process_and_show_mask(
    mask: np.ndarray, ax: plt.Axes, obj_id: Optional[int] = None, random_color: bool = False, borders: bool = False
) -> np.ndarray:
    # Generate mask color
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        if obj_id is not None:
            cmap = plt.get_cmap("tab10")
            color = np.array([*cmap(obj_id)[:3], 0.6])
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    # Process mask
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Add borders if requested
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

    ax.imshow(mask_image)
    return mask_image


def show_points(
    coords: np.ndarray, labels: np.ndarray, ax: plt.Axes, marker_size: int = 375
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Display point prompts and return point coordinates for testing.
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

    return pos_points, neg_points


def show_box(box: np.ndarray, ax: plt.Axes) -> Tuple[float, float, float, float]:
    """
    Display a bounding box and return its coordinates for testing.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    return x0, y0, w, h
