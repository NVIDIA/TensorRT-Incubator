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

from typing import List, Optional, Tuple

import tripy as tp

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class Dummy(tp.Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class MaskDecoder(tp.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: tp.Module,
        num_multimask_outputs: int = 3,
        activation=tp.gelu,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        dtype=tp.float32,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.dtype = dtype

        self.num_multimask_outputs = num_multimask_outputs
        self.activation = activation

        self.iou_token = tp.Embedding(1, transformer_dim, dtype=dtype)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = tp.Embedding(self.num_mask_tokens, transformer_dim, dtype=dtype)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = tp.Embedding(1, transformer_dim, dtype=dtype)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = [
            tp.ConvTranspose(
                transformer_dim,
                transformer_dim // 4,
                kernel_dims=(2, 2),
                stride=(2, 2),
                dtype=dtype,
            ),
            LayerNorm2d(transformer_dim // 4),
            Dummy(),  # Accounts for Dropout layer, needed for weight loading
            tp.ConvTranspose(
                transformer_dim // 4,
                transformer_dim // 8,
                kernel_dims=(2, 2),
                stride=(2, 2),
                dtype=dtype,
            ),
            Dummy(),  # Accounts for Dropout layer, needed for weight loading
        ]
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = tp.Conv(
                transformer_dim,
                transformer_dim // 8,
                kernel_dims=(1, 1),
                stride=(1, 1),
                dtype=dtype,
            )
            self.conv_s1 = tp.Conv(
                transformer_dim,
                transformer_dim // 4,
                kernel_dims=(1, 1),
                stride=(1, 1),
                dtype=dtype,
            )

        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3, dtype=dtype)
            for i in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
            dtype=dtype,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = tp.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3, dtype=dtype)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def __call__(
        self,
        image_embeddings: tp.Tensor,
        image_pe: tp.Tensor,
        sparse_prompt_embeddings: tp.Tensor,
        dense_prompt_embeddings: tp.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features_1: Optional[tp.Tensor] = None,
        high_res_features_2: Optional[tp.Tensor] = None,
    ) -> Tuple[tp.Tensor, tp.Tensor]:

        return self.forward(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output,
            repeat_image,
            high_res_features_1,
            high_res_features_2,
        )

    def forward(
        self,
        image_embeddings: tp.Tensor,
        image_pe: tp.Tensor,
        sparse_prompt_embeddings: tp.Tensor,
        dense_prompt_embeddings: tp.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features_1: Optional[tp.Tensor] = None,
        high_res_features_2: Optional[tp.Tensor] = None,
    ) -> Tuple[tp.Tensor, tp.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (tp.Tensor): the embeddings from the image encoder
          image_pe (tp.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (tp.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (tp.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          tp.Tensor: batched predicted masks
          tp.Tensor: batched predictions of mask quality
          tp.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features_1=high_res_features_1,
            high_res_features_2=high_res_features_2,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
            # masks = masks[:, 0:1, :, :]
            # iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: tp.Tensor,
        image_pe: tp.Tensor,
        sparse_prompt_embeddings: tp.Tensor,
        dense_prompt_embeddings: tp.Tensor,
        repeat_image: bool,
        high_res_features_1: Optional[tp.Tensor] = None,
        high_res_features_2: Optional[tp.Tensor] = None,
    ) -> Tuple[tp.Tensor, tp.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = tp.concatenate(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = tp.concatenate([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = tp.expand(tp.unsqueeze(output_tokens, 0), (sparse_prompt_embeddings.shape[0], -1, -1))
        tokens = tp.concatenate((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = tp.repeat(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings

        src = src + dense_prompt_embeddings
        pos_src = tp.repeat(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        og_dtype = src.dtype
        src = tp.cast(src, self.dtype)
        pos_src = tp.cast(pos_src, self.dtype)
        tokens = tp.cast(tokens, self.dtype)
        hs, src = self.transformer(src, pos_src, tokens)
        hs = tp.cast(hs, og_dtype)
        src = tp.cast(src, og_dtype)

        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : s + 1 + self.num_mask_tokens, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = tp.reshape(tp.transpose(src, 1, 2), (b, c, h, w))
        act1 = self.activation
        act2 = self.activation

        if not self.use_high_res_features:
            dc1, ln1, _, dc2, _ = self.output_upscaling
            post_ln1 = tp.cast(ln1(tp.cast(dc1(src), tp.float32)), src.dtype)
            upscaled_embedding = act2(dc2(act1(post_ln1)))
            # upscaled_embedding = act2(dc2(act1(ln1(dc1(src)))))
        else:
            dc1, ln1, _, dc2, _ = self.output_upscaling
            feat_s0, feat_s1 = high_res_features_1, high_res_features_2
            post_ln1 = tp.cast(ln1(tp.cast(dc1(src) + feat_s1, tp.float32)), src.dtype)
            upscaled_embedding = act1(post_ln1)
            # upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[tp.Tensor] = []
        for i in range(self.num_mask_tokens):
            mlp_in = mask_tokens_out[:, i, :]
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mlp_in))
        hyper_in = tp.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        upscaled_embedding = tp.reshape(upscaled_embedding, (b, c, h * w))
        # out_4 = upscaled_embedding
        masks = tp.reshape(hyper_in @ upscaled_embedding, (b, -1, h, w))

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * tp.ones(
                (iou_pred.shape[0], 1), dtype=self.dtype
            )  # iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds, similar to https://github.com/fairinternal/onevision/pull/568.
        """
        mask_logits = tp.flatten(mask_logits, -2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = tp.cast(tp.sum(tp.cast(mask_logits > stability_delta, tp.float32), dim=-1), self.dtype)
        area_u = tp.cast(tp.sum(tp.cast(mask_logits > -stability_delta, tp.float32), dim=-1), self.dtype)
        stability_scores = tp.where(area_u > 0, area_i / area_u, tp.Tensor(1.0, dtype=self.dtype))
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = tp.argmax(multimask_iou_scores, dim=-1)
        batch_inds = tp.arange(multimask_iou_scores.shape[0], dtype=self.dtype)
        batch_inds = tp.cast(batch_inds, tp.int32)

        def advanced_indexing(tensor, first_index, second_index):
            # Step 1: Use the first_index to select rows
            step1 = tp.gather(tensor, dim=0, index=first_index)

            # Step 2: Prepare for the second gather operation
            batch_size = first_index.shape[0]
            row_indices = tp.arange(batch_size, dtype=tp.int32)

            # Step 3: Combine row_indices and second_index
            combined_indices = tp.stack([row_indices, second_index], dim=1)

            # Step 4: Flatten the tensor
            flattened = tp.flatten(step1)

            # Step 5: Calculate flat indices
            flat_indices = combined_indices[:, 0] * batch_size + combined_indices[:, 1]

            # Step 6: Gather using flat indices
            result = tp.gather(flattened, dim=0, index=flat_indices)

            return result

        best_multimask_logits = advanced_indexing(multimask_logits, batch_inds, best_scores_inds)
        best_multimask_iou_scores = advanced_indexing(multimask_iou_scores, batch_inds, best_scores_inds)

        best_multimask_logits = tp.unsqueeze(best_multimask_logits, 1)
        best_multimask_iou_scores = tp.unsqueeze(best_multimask_iou_scores, 1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        is_stable = tp.unsqueeze(tp.unsqueeze(is_stable, -1), -1)
        mask_logits_out = tp.where(
            is_stable,
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = tp.where(
            is_stable,
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
