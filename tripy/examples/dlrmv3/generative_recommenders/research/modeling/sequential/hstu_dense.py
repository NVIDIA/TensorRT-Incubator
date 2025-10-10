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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Dense-only implementation of HSTU (Hierarchical Sequential Transduction Unit).
This version eliminates all jagged tensor operations and works entirely with dense tensors,
using attention masks to handle variable-length sequences.
"""

import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.research.modeling.sequential.utils import (
    get_current_embeddings,
)
from generative_recommenders.research.modeling.similarity_module import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from generative_recommenders.research.rails.similarities.module import SimilarityModule

from generative_recommenders.research.modeling.sequential.hstu import (
    RelativeAttentionBiasModule,
    RelativePositionalBias,
    RelativeBucketedTimeAndPositionBasedBias,
)


TIMESTAMPS_KEY = "timestamps"


HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _hstu_attention_maybe_from_cache_dense(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,  # [B, N, num_heads * attention_dim]
    k: torch.Tensor,  # [B, N, num_heads * attention_dim]
    v: torch.Tensor,  # [B, N, num_heads * linear_dim]
    cached_q: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    all_timestamps: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,  # [B, N, N] or [N, N]
    rel_attn_bias: RelativeAttentionBiasModule,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dense attention computation without jagged tensor conversions.
    Padded positions (zeros) are handled naturally by the operations.

    Args:
        q, k, v: Dense tensors [B, N, D] (padded with zeros beyond sequence length)
        invalid_attn_mask: [B, N, N] or [N, N] causal mask

    Returns:
        attention_output: [B, N, num_heads * linear_dim]
        q: [B, N, num_heads * attention_dim] (for caching)
        k: [B, N, num_heads * attention_dim] (for caching)
    """
    B, N = q.shape[:2]

    if verbose:
        print(f"          🧠 Dense attention: Q,K,V shapes {q.shape}, {k.shape}, {v.shape}")

    # Handle caching (incremental updates)
    if cached_q is not None and cached_k is not None:
        if verbose:
            print(f"          🔄 Using cached Q,K (incremental mode)")
        # For incremental updates, we would update only specific positions
        # This is a simplified version - full caching would need delta_offsets handling
        padded_q, padded_k = cached_q, cached_k
    else:
        padded_q, padded_k = q, k

    # Reshape for multi-head attention: [B, N, H, D]
    q_heads = padded_q.view(B, N, num_heads, attention_dim)
    k_heads = padded_k.view(B, N, num_heads, attention_dim)
    v_heads = v.view(B, N, num_heads, linear_dim)

    # Compute attention scores: [B, H, N, N]
    qk_attn = torch.einsum("bnhd,bmhd->bhnm", q_heads, k_heads)

    # Add relative attention bias if available
    if all_timestamps is not None:
        if verbose:
            print(f"          📍 Applying relative attention bias")
        qk_attn = qk_attn + rel_attn_bias(all_timestamps).unsqueeze(1)  # [B, 1, N, N] -> [B, H, N, N]

    # Apply SiLU activation and scaling (same as original)
    qk_attn = F.silu(qk_attn) / N

    # Apply causal mask - handle both [N, N] and [B, N, N] cases (same as original)
    if invalid_attn_mask.dim() == 2:
        # [N, N] -> [B, H, N, N]
        qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    else:
        # [B, N, N] -> [B, H, N, N]
        qk_attn = qk_attn * invalid_attn_mask.unsqueeze(1)

    # Compute attention output: [B, H, N, N] x [B, N, H, D] -> [B, N, H, D]
    # Note: Padded positions in v are zero, so attention to them produces zero
    attention_result = torch.einsum("bhnm,bmhd->bnhd", qk_attn, v_heads)

    # Reshape back: [B, N, H*D]
    attention_output = attention_result.reshape(B, N, num_heads * linear_dim)

    if verbose:
        print(f"          ✅ Dense attention complete: {attention_output.shape}")

    return attention_output, padded_q, padded_k


class SequentialTransductionUnitDense(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = relative_attention_bias_module
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 2 * num_heads + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._verbose: bool = verbose
        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads * (3 if concat_ua else 1),
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps)

    def forward(
        self,
        x: torch.Tensor,  # [B, N, D] (padded with zeros beyond sequence length)
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,  # [B, N, N] or [N, N]
        cached_q: Optional[torch.Tensor] = None,
        cached_k: Optional[torch.Tensor] = None,
        cached_v: Optional[torch.Tensor] = None,
        cached_outputs: Optional[torch.Tensor] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[HSTUCacheState]]:
        """
        Dense forward pass without jagged tensor operations.
        Padded positions (zeros) are handled naturally by the operations.

        Args:
            x: [B, N, D] input embeddings (padded with zeros beyond sequence length)
            invalid_attn_mask: [B, N, N] or [N, N] causal attention mask

        Returns:
            output: [B, N, D] (padded positions remain zero)
            cache_state: Optional cache for incremental inference
        """
        B, N, D = x.shape

        if self._verbose:
            print(f"        🧠 STU Dense Layer: {x.shape}")

        # Layer normalization on input (zeros remain zeros after normalization)
        normed_x = self._norm_input(x)

        # Linear projection to get u, v, q, k
        # Note: For padded positions (zeros), this produces zeros
        if self._linear_config == "uvqk":
            # Reshape for batch matrix multiply: [B*N, D] @ [D, 4*H*D'] -> [B*N, 4*H*D']
            normed_x_flat = normed_x.view(-1, self._embedding_dim)
            batched_mm_output = torch.mm(normed_x_flat, self._uvqk)
            batched_mm_output = batched_mm_output.view(B, N, -1)

            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                pass

            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        # Handle caching for incremental inference
        if cached_v is not None:
            # For incremental updates - this is simplified
            v = cached_v  # In practice, would need proper incremental update logic

        # Attention computation
        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            assert self._rel_attn_bias is not None
            if self._verbose:
                print(f"          🎯 Dense attention with relative bias ({self._normalization})")
            attn_output, cached_q_new, cached_k_new = _hstu_attention_maybe_from_cache_dense(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                cached_q=cached_q,
                cached_k=cached_k,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=self._rel_attn_bias,
                verbose=self._verbose,
            )
        elif self._normalization == "softmax_rel_bias":
            # Standard softmax attention (not used in main HSTU)
            q_heads = q.view(B, N, self._num_heads, self._attention_dim)
            k_heads = k.view(B, N, self._num_heads, self._attention_dim)
            v_heads = v.view(B, N, self._num_heads, self._linear_dim)

            # [B, H, N, N]
            qk_attn = torch.einsum("bnhd,bmhd->bhnm", q_heads, k_heads)
            if self._rel_attn_bias is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps).unsqueeze(1)
            qk_attn = F.softmax(qk_attn / math.sqrt(self._attention_dim), dim=-1)

            # Apply causal mask
            if invalid_attn_mask.dim() == 2:
                qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                qk_attn = qk_attn * invalid_attn_mask.unsqueeze(1)

            attn_output = torch.einsum("bhnm,bmhd->bnhd", qk_attn, v_heads)
            attn_output = attn_output.reshape(B, N, self._num_heads * self._linear_dim)

            cached_q_new, cached_k_new = q, k
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        # Combine u and attention output
        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        # Final linear projection with residual connection
        # Note: Padded positions remain zero due to zero inputs
        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + x
        )

        # Handle caching
        cache_state = None
        if return_cache_states:
            cache_state = (v, cached_q_new, cached_k_new, new_outputs)

        return new_outputs, cache_state


class HSTUDense(torch.nn.Module):
    def __init__(
        self,
        modules: List[SequentialTransductionUnitDense],
        autocast_dtype: Optional[torch.dtype],
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=modules)
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype
        self._verbose: bool = verbose

    def forward(
        self,
        x: torch.Tensor,  # [B, N, D] (padded with zeros beyond sequence length)
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,  # [B, N, N] or [N, N]
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Dense forward pass through all attention layers.
        Padded positions (zeros) are handled naturally by the operations.

        Args:
            x: [B, N, D] input embeddings (padded with zeros beyond sequence length)
            invalid_attn_mask: [B, N, N] or [N, N] causal attention mask

        Returns:
            output: [B, N, D] (padded positions remain zero)
            cache_states: List of cache states for each layer
        """
        if self._verbose:
            print(f"    🏗️  HSTUDense processing {len(self._attention_layers)} layers...")

        cache_states: List[HSTUCacheState] = []

        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                if self._verbose:
                    print(f"      🧠 Layer {i+1}/{len(self._attention_layers)}")
                    print(f"        Input to layer {i+1}: x[0,0,:5] = {x[0,0,:5]}")

                layer_cache = cache[i] if cache is not None else None
                cached_v = layer_cache[0] if layer_cache is not None else None
                cached_q = layer_cache[1] if layer_cache is not None else None
                cached_k = layer_cache[2] if layer_cache is not None else None
                cached_outputs = layer_cache[3] if layer_cache is not None else None

                x, cache_state = layer(
                    x=x,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    cached_q=cached_q,
                    cached_k=cached_k,
                    cached_v=cached_v,
                    cached_outputs=cached_outputs,
                    return_cache_states=return_cache_states,
                )

                if self._verbose:
                    print(f"        Output from layer {i+1}: x[0,0,:5] = {x[0,0,:5]}")

                if return_cache_states:
                    cache_states.append(cache_state)

        return x, cache_states


class HSTUDenseModel(SequentialEncoderWithLearnedSimilarityModule):
    """
    Dense-only implementation of HSTU that eliminates all jagged tensor operations.
    Uses attention masks to handle variable-length sequences instead.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_config: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = input_features_preproc_module
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias

        self._hstu = HSTUDense(
            modules=[
                SequentialTransductionUnitDense(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_sequence_len + max_output_len,
                            num_buckets=128,
                            bucketization_fn=lambda x: (torch.log(torch.abs(x).clamp(min=1)) / 0.301).long(),
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    concat_ua=concat_ua,
                    verbose=verbose,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
            verbose=verbose,
        )

        # Causal attention mask
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length + max_output_len,
                        self._max_sequence_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = verbose
        self.reset_params()

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if ("_hstu" in name) or ("_embedding_module" in name):
                if self._verbose:
                    print(f"Skipping init for {name}")
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(f"Initialize {name} as xavier normal: {params.data.size()} params")
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        debug_str = (
            f"HSTUDense-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Dense version of user embedding generation.
        Input embeddings are already padded with zeros beyond sequence length.

        Args:
            past_lengths: [B] sequence lengths
            past_embeddings: [B, N, D] dense embeddings (padded with zeros)

        Returns:
            user_embeddings: [B, N, D] (padded positions remain zero)
            cache_states: List of cache states
        """
        device = past_lengths.device
        float_dtype = past_embeddings.dtype
        B, N, _ = past_embeddings.size()

        if self._verbose:
            print(f"  📊 Dense input preprocessing...")
            print(f"    Input embeddings shape: {past_embeddings.shape}")
            print(f"    Attention mask shape: {self._attn_mask.shape}")

        # Input preprocessing (same as original)
        # Note: This produces padded dense tensors with zeros beyond sequence length
        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        if self._verbose:
            print(f"    After preprocessing shape: {user_embeddings.shape}")
            print(f"    Preprocessing sample [0,0,:5]: {user_embeddings[0,0,:5]}")
            print(f"  🔢 Dense HSTU forward pass...")

        # Dense HSTU forward pass
        # Pad to attention mask size to match behavior when converting from dense -> jagged -> dense
        attention_mask_size = self._attn_mask.size(-1)
        current_size = user_embeddings.size(1)

        if current_size < attention_mask_size:
            # Pad to match attention mask size, just like jagged_to_padded_dense does
            padding_size = attention_mask_size - current_size
            user_embeddings = torch.nn.functional.pad(
                user_embeddings, (0, 0, 0, padding_size), value=0.0
            )  # Pad sequence dimension with zeros
            if self._verbose:
                print(f"    Padded to attention mask size: {user_embeddings.shape}")
                print(f"    Padded sample [0,0,:5]: {user_embeddings[0,0,:5]}")

        user_embeddings, cached_states = self._hstu(
            x=user_embeddings,
            all_timestamps=(past_payloads[TIMESTAMPS_KEY] if TIMESTAMPS_KEY in past_payloads else None),
            invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
            cache=cache,
            return_cache_states=return_cache_states,
        )

        if self._verbose:
            print(f"    After HSTU layers: {user_embeddings.shape}")
            print(f"    HSTU output sample [0,0,:5]: {user_embeddings[0,0,:5]}")

        # Zero out positions beyond actual sequence lengths to match jagged behavior
        # The jagged implementation naturally excludes these positions from computation
        for i, length in enumerate(past_lengths):
            if length < user_embeddings.size(1):
                if self._verbose:
                    print(
                        f"    PyTorch: Zeroing batch {i} positions {length}:{user_embeddings.size(1)} (length={length})"
                    )
                user_embeddings[i, length:] = 0.0

        if self._verbose:
            print(f"    After zeroing padded positions: {user_embeddings.shape}")
            print(f"    After zeroing sample [0,0,:5]: {user_embeddings[0,0,:5]}")

        # Output postprocessing
        result = self._output_postproc(user_embeddings)

        if self._verbose:
            print(f"    After postprocessing: {result.shape}")
            print(f"    Final sample [0,0,:5]: {result[0,0,:5]}")

        return result, cached_states

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Dense forward pass."""
        if self._verbose:
            print(f"🚀 HSTUDense.forward() → generate_user_embeddings()")

        encoded_embeddings, _ = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        if self._verbose:
            print(f"✅ HSTUDense.forward() complete: {encoded_embeddings.shape}")

        return encoded_embeddings

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        cache: Optional[List[HSTUCacheState]],
        return_cache_states: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """Dense encoding to get current embeddings."""
        encoded_seq_embeddings, cache_states = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        current_embeddings = get_current_embeddings(lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings)
        if return_cache_states:
            return current_embeddings, cache_states
        else:
            return current_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """Dense encoding interface."""
        if self._verbose:
            print(f"🎯 HSTUDense.encode() → extracting current embeddings")

        result = self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            cache=cache,
            return_cache_states=return_cache_states,
        )

        if self._verbose:
            if return_cache_states:
                print(f"✅ HSTUDense.encode() complete: {result[0].shape}")
            else:
                print(f"✅ HSTUDense.encode() complete: {result.shape}")

        return result
