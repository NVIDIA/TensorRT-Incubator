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

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import tripy as tp

from PIL.Image import Image

from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms


class SAM2ImagePredictor:
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        **kwargs,
    ) -> None:
        """
        Compute image embedding for a given image and then perform mask prediction using the user provided prompt.
        """
        super().__init__()
        self.model = sam_model
        self.device = torch.device("cuda")

        # Transforms using torch
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])
        assert len(set(self._orig_hw)) == 1, "Images in the batch must have the same size."
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        img_batch_tp = tp.Tensor(img_batch.to(getattr(torch, self.model.image_encoder.trunk.dtype)).contiguous())
        backbone_out = self.model.forward_image(img_batch_tp)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed.to(vision_feats[-1].dtype)

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Predict masks for the given input prompts, using the currently set images.

        Arguments:
          point_coords_batch: A list of Nx2 arrays of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels_batch: A list of length N arrays of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          multimask_output: If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits: If true, returns un-thresholded masks logits
            instead of a binary mask.
          normalize_coords: If true, the point coordinates will be normalized to
            the range [0,1] and point_coords is expected to be wrt. image dimensions.

        Returns:
          masks: A list of output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          ious: A list of arrays of length C containing the model's
            predictions for the quality of each mask.
          low_res_masks: A list of arrays of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image_batch(...) before mask prediction.")

        def concat_batch(x):
            if x is None:
                return x
            return np.concatenate(x, axis=0)

        point_coords = concat_batch(point_coords_batch)
        point_labels = concat_batch(point_labels_batch)

        _, unnorm_coords, labels, _ = self._prep_prompts(
            point_coords,
            point_labels,
            None,  # box
            None,  # mask_input
            normalize_coords,
        )
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            multimask_output,
            return_logits=return_logits,
        )

        def to_np_list(x):
            x = x.float().detach().cpu().numpy()
            return [xi for xi in x]

        all_masks = to_np_list(masks)
        all_ious = to_np_list(iou_predictions)
        all_low_res_masks = to_np_list(low_res_masks)

        return all_masks, all_ious, all_low_res_masks

    def _prep_prompts(self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1):
        """
        point_coords: [B, 2] -> [B, 1, 2]
        point_labels: [B] -> [B, 1]
        """
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords.unsqueeze(1), labels.unsqueeze(1)
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(mask_logits, dtype=torch.float, device=self.device)
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points_x=tp.Tensor(point_coords.contiguous()),
            points_y=tp.Tensor(point_labels.contiguous()),
        )

        # Predict masks
        self.dense_pe = self.model.sam_prompt_encoder.get_dense_pe()
        image_embedding = self._features["image_embed"].contiguous()
        high_res_features_1 = self._features["high_res_feats"][0].contiguous()
        high_res_features_2 = self._features["high_res_feats"][1].contiguous()

        if self.model.model_dtype == tp.float16:
            image_embedding = image_embedding.half()
            high_res_features_1 = high_res_features_1.half()
            high_res_features_2 = high_res_features_2.half()

        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder_true(
            image_embeddings=tp.Tensor(image_embedding),
            image_pe=self.dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features_1=tp.Tensor(high_res_features_1),
            high_res_features_2=tp.Tensor(high_res_features_2),
        )
        low_res_masks = torch.from_dlpack(low_res_masks)
        iou_predictions = torch.from_dlpack(iou_predictions)

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(low_res_masks, self._orig_hw[img_idx])
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
