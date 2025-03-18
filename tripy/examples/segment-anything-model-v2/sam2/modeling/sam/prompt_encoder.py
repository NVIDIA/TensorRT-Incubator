# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling SAM2 with Tripy or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
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

from typing import Optional, Tuple, Type

import nvtripy as tp
from sam2.modeling.position_encoding import PositionEmbeddingRandom
from sam2.modeling.sam2_utils import LayerNorm2d


class PromptEncoder(tp.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[tp.Module] = tp.gelu,
        dtype=tp.float32,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (tp.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        self.point_embeddings = [tp.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.not_a_point_embed = tp.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = [
            tp.Conv(1, mask_in_chans // 4, kernel_dims=(2, 2), stride=(2, 2)),
            LayerNorm2d(mask_in_chans // 4),
            activation,
            tp.Conv(mask_in_chans // 4, mask_in_chans, kernel_dims=(2, 2), stride=(2, 2)),
            LayerNorm2d(mask_in_chans),
            activation,
            tp.Conv(mask_in_chans, embed_dim, kernel_dims=(1, 1)),
        ]
        self.no_mask_embed = tp.Embedding(1, embed_dim)
        self.dtype = dtype

    def get_dense_pe(self) -> tp.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          tp.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        dense_pe = tp.unsqueeze(self.pe_layer(self.image_embedding_size), 0)
        return tp.cast(dense_pe, self.dtype)

    def _embed_points(
        self,
        points: tp.Tensor,
        labels: tp.Tensor,
        pad: bool,
    ) -> tp.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = tp.zeros((points.shape[0], 1, 2), dtype=points.dtype)
            padding_label = 0 - tp.ones((labels.shape[0], 1), dtype=labels.dtype)
            padding_label = tp.cast(padding_label, labels.dtype)
            points = tp.concatenate([points, padding_point], dim=1)
            labels = tp.concatenate([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        labels = tp.unsqueeze(labels, 2)
        point_embedding = tp.where(labels == -1, tp.Tensor([0.0]), point_embedding)
        point_embedding = tp.where(
            labels == -1,
            point_embedding + self.not_a_point_embed.weight,
            point_embedding,
        )
        point_embedding = tp.where(
            labels == 0,
            point_embedding + self.point_embeddings[0].weight,
            point_embedding,
        )
        point_embedding = tp.where(
            labels == 1,
            point_embedding + self.point_embeddings[1].weight,
            point_embedding,
        )
        point_embedding = tp.where(
            labels == 2,
            point_embedding + self.point_embeddings[2].weight,
            point_embedding,
        )
        point_embedding = tp.where(
            labels == 3,
            point_embedding + self.point_embeddings[3].weight,
            point_embedding,
        )
        return point_embedding

    def _embed_boxes(self, boxes: tp.Tensor) -> tp.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = tp.reshape(boxes, (-1, 2, 2))
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)

        corner_embedding_0 = corner_embedding[:, 0, :] + self.point_embeddings[2].weight
        corner_embedding_1 = corner_embedding[:, 1, :] + self.point_embeddings[3].weight

        # Combine the updated x and y coordinates into a new tensor
        new_corner_embedding = tp.stack([corner_embedding_0, corner_embedding_1], dim=1)

        return new_corner_embedding

    def _embed_masks(self, masks: tp.Tensor) -> tp.Tensor:
        """Embeds mask inputs."""
        mask_embedding = masks
        for l in self.mask_downscaling:
            mask_embedding = l(mask_embedding)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[tp.Tensor, tp.Tensor]],
        boxes: Optional[tp.Tensor],
        masks: Optional[tp.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def forward(
        self,
        points_x: Optional[tp.Tensor],
        points_y: Optional[tp.Tensor],
        boxes: Optional[tp.Tensor],
        masks: Optional[tp.Tensor],
    ) -> Tuple[tp.Tensor, tp.Tensor]:
        return self.forward_impl(points_x, points_y, boxes, masks)

    def forward_impl(
        self,
        points_x: Optional[tp.Tensor],
        points_y: Optional[tp.Tensor],
        boxes: Optional[tp.Tensor],
        masks: Optional[tp.Tensor],
    ) -> Tuple[tp.Tensor, tp.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(tp.Tensor, tp.Tensor) or none): point coordinates
            and labels to embed.
          boxes (tp.Tensor or none): boxes to embed
          masks (tp.Tensor or none): masks to embed

        Returns:
          tp.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          tp.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        points = (points_x, points_y)
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = tp.zeros((bs, 0, self.embed_dim))
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = tp.concatenate([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = tp.concatenate([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = tp.reshape(self.no_mask_embed.weight, (1, -1, 1, 1))
            dense_embeddings = tp.expand(
                dense_embeddings,
                (bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]),
            )

        sparse_embeddings = tp.cast(sparse_embeddings, self.dtype)
        dense_embeddings = tp.cast(dense_embeddings, self.dtype)
        return sparse_embeddings, dense_embeddings
