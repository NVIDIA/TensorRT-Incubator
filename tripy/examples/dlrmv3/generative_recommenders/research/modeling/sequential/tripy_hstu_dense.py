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
import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch

import nvtripy as tp
from nvtripy.frontend.module.parameter import DefaultParameter

from generative_recommenders.research.modeling.sequential.tripy_embedding_modules import (
    TripyEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.tripy_input_features_preprocessors import (
    TripyInputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.tripy_output_postprocessors import (
    TripyOutputPostprocessorModule,
)
from generative_recommenders.research.modeling.sequential.tripy_utils import (
    tripy_get_current_embeddings,
)
from generative_recommenders.research.modeling.tripy_similarity_module import (
    TripySequentialEncoderWithLearnedSimilarityModule,
    TripySimilarityModule,
)


TIMESTAMPS_KEY = "timestamps"


class RelativeAttentionBiasModule(tp.Module):
    @abc.abstractmethod
    def forward(self, all_timestamps: tp.Tensor) -> tp.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            float tensor broadcastable to [B, N, N]
        """
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self._max_seq_len = max_seq_len
        self._w = DefaultParameter((2 * max_seq_len - 1,), dtype=tp.float32)
        self._w = tp.Tensor(torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02))

    def forward(self, all_timestamps: tp.Tensor) -> tp.Tensor:
        # all_timestamps is ignored
        n = self._max_seq_len
        # Pad self._w to length 2*n-1 + n zeros at the end
        w_padded = tp.pad(self._w[: 2 * n - 1], [(0, n)])
        # Repeat the whole padded vector n times (concatenate n copies)
        t = tp.concatenate([w_padded] * n, dim=0)
        t = t[:-n]
        t = tp.reshape(t, (1, n, 3 * n - 2))
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(self, max_seq_len: int, num_buckets: int, bucketization_fn) -> None:
        super().__init__()

        self._max_seq_len = max_seq_len
        self._num_buckets = num_buckets
        self._ts_w = DefaultParameter((num_buckets + 1,), dtype=tp.float32)
        self._pos_w = DefaultParameter((2 * max_seq_len - 1,), dtype=tp.float32)
        self._ts_w = tp.Tensor(torch.empty(num_buckets + 1).normal_(mean=0, std=0.02))
        self._pos_w = tp.Tensor(torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02))
        self._bucketization_fn = bucketization_fn

    def forward(self, all_timestamps: tp.Tensor) -> tp.Tensor:
        B = all_timestamps.shape[0]
        N = self._max_seq_len
        # Pad pos_w to length 2*N-1 + N zeros at the end
        t = tp.pad(self._pos_w[: 2 * N - 1], [(0, N)])
        t = tp.concatenate([t] * N, dim=0)
        t = t[:-N]
        t = tp.reshape(t, (1, N, 3 * N - 2))
        r = (2 * N - 1) // 2
        rel_pos_bias = t[..., r:-r]
        # [B, N+1] for easier manipulation
        ext_timestamps = tp.concatenate([all_timestamps, all_timestamps[:, N - 1 : N]], dim=1)
        # bucketed_timestamps: [B, N, N]
        bucketed = self._bucketization_fn(
            tp.unsqueeze(ext_timestamps[:, 1:], 2) - tp.unsqueeze(ext_timestamps[:, :-1], 1)
        )
        bucketed_timestamps = tp.minimum(tp.maximum(bucketed, 0), self._num_buckets)
        # rel_ts_bias: [B, N, N]
        rel_ts_bias = tp.gather(self._ts_w, 0, bucketed_timestamps.reshape((-1,))).reshape((B, N, N))
        return rel_pos_bias + rel_ts_bias


HSTUCacheState = Tuple[tp.Tensor, tp.Tensor, tp.Tensor, tp.Tensor]


def _tripy_hstu_attention_maybe_from_cache_dense(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: tp.Tensor,  # [B, N, num_heads * attention_dim]
    k: tp.Tensor,  # [B, N, num_heads * attention_dim]
    v: tp.Tensor,  # [B, N, num_heads * linear_dim]
    cached_q: Optional[tp.Tensor],
    cached_k: Optional[tp.Tensor],
    all_timestamps: Optional[tp.Tensor],
    invalid_attn_mask: tp.Tensor,  # [B, N, N] or [N, N]
    rel_attn_bias: RelativeAttentionBiasModule,
    verbose: bool = False,
) -> Tuple[tp.Tensor, tp.Tensor, tp.Tensor]:
    """
    Tripy attention computation without jagged tensor conversions.
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
        print(f"          🧠 Tripy attention: Q,K,V shapes {q.shape}, {k.shape}, {v.shape}")

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
    q_heads = tp.reshape(padded_q, (B, N, num_heads, attention_dim))
    k_heads = tp.reshape(padded_k, (B, N, num_heads, attention_dim))
    v_heads = tp.reshape(v, (B, N, num_heads, linear_dim))

    # Compute attention scores: [B, H, N, N]
    # Tripy equivalent of torch.einsum("bnhd,bmhd->bhnm", q_heads, k_heads)
    qk_attn = tp.permute(q_heads, (0, 2, 1, 3)) @ tp.permute(  # [B, H, N, D]
        k_heads, (0, 2, 3, 1)
    )  # [B, H, D, N]  # [B, H, N, N]

    # Add relative attention bias if available
    if all_timestamps is not None:
        if verbose:
            print(f"          📍 Applying relative attention bias")
        rel_bias = rel_attn_bias(all_timestamps)  # [B, N, N]
        qk_attn = qk_attn + tp.unsqueeze(rel_bias, 1)  # [B, 1, N, N] -> [B, H, N, N]

    # Apply SiLU activation and scaling (same as original)
    qk_attn = tp.silu(qk_attn) / N.cast(tp.float32)

    # Apply causal mask - handle both [N, N] and [B, N, N] cases (same as original)
    if len(invalid_attn_mask.shape) == 2:
        # [N, N] -> [B, H, N, N]
        qk_attn = qk_attn * tp.unsqueeze(tp.unsqueeze(invalid_attn_mask, 0), 0).cast(tp.float32)
    else:
        # [B, N, N] -> [B, H, N, N]
        qk_attn = qk_attn * tp.unsqueeze(invalid_attn_mask, 1).cast(tp.float32)

    # Compute attention output: [B, H, N, N] x [B, N, H, D] -> [B, N, H, D]
    # Tripy equivalent of torch.einsum("bhnm,bmhd->bnhd", qk_attn, v_heads)
    v_heads_transposed = tp.permute(v_heads, (0, 2, 1, 3))  # [B, H, N, D]
    attention_result = qk_attn @ v_heads_transposed  # [B, H, N, D]
    attention_result = tp.permute(attention_result, (0, 2, 1, 3))  # [B, N, H, D]

    # Reshape back: [B, N, H*D]
    attention_output = tp.reshape(attention_result, (B, N, num_heads * linear_dim))

    if verbose:
        print(f"          ✅ Tripy attention complete: {attention_output.shape}")

    return attention_output, padded_q, padded_k


class TripySequentialTransductionUnitDense(tp.Module):
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
            uvqk_shape = (
                embedding_dim,
                linear_hidden_dim * 2 * num_heads + attention_dim * num_heads * 2,
            )
            self._uvqk = DefaultParameter(uvqk_shape, dtype=tp.float32)
            self._uvqk = tp.Tensor(torch.empty(uvqk_shape).normal_(mean=0, std=0.02))
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._verbose: bool = verbose

        out_features = embedding_dim
        in_features = linear_hidden_dim * num_heads * (3 if concat_ua else 1)
        self._o_weight = DefaultParameter((in_features, out_features), dtype=tp.float32)
        self._o_bias = DefaultParameter((embedding_dim,), dtype=tp.float32)
        self._o_weight = tp.Tensor(
            torch.empty(in_features, out_features).uniform_(
                -math.sqrt(6.0 / (in_features + out_features)), math.sqrt(6.0 / (in_features + out_features))
            )
        )
        self._o_bias = tp.zeros((embedding_dim,))

        self._eps: float = epsilon

    # TODO: use functional layernorm with no affine transform
    def _norm_input(self, x: tp.Tensor) -> tp.Tensor:
        my_norm = tp.LayerNorm(self._embedding_dim, eps=self._eps)
        my_norm.weight = tp.ones((self._embedding_dim,))
        my_norm.bias = tp.zeros((self._embedding_dim,))
        return my_norm(x)

    def _norm_attn_output(self, x: tp.Tensor) -> tp.Tensor:
        my_norm = tp.LayerNorm(self._linear_dim * self._num_heads, eps=self._eps)
        my_norm.weight = tp.ones((self._linear_dim * self._num_heads,))
        my_norm.bias = tp.zeros((self._linear_dim * self._num_heads,))
        return my_norm(x)

    def forward(
        self,
        x: tp.Tensor,  # [B, N, D] (padded with zeros beyond sequence length)
        all_timestamps: Optional[tp.Tensor],
        invalid_attn_mask: tp.Tensor,  # [B, N, N] or [N, N]
        cached_q: Optional[tp.Tensor] = None,
        cached_k: Optional[tp.Tensor] = None,
        cached_v: Optional[tp.Tensor] = None,
        cached_outputs: Optional[tp.Tensor] = None,
        return_cache_states: bool = False,
    ) -> Tuple[tp.Tensor, Optional[HSTUCacheState]]:
        """
        Tripy forward pass without jagged tensor operations.
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
            print(f"        🧠 STU Tripy Layer: {x.shape}")

        # Layer normalization on input (zeros remain zeros after normalization)
        normed_x = self._norm_input(x)

        # Linear projection to get u, v, q, k
        # Note: For padded positions (zeros), this produces zeros
        if self._linear_config == "uvqk":
            # Tripy matrix multiply: [B, N, D] @ [D, 4*H*D'] -> [B, N, 4*H*D']
            batched_mm_output = normed_x @ self._uvqk

            if self._linear_activation == "silu":
                batched_mm_output = tp.silu(batched_mm_output)
            elif self._linear_activation == "none":
                pass

            # Split into u, v, q, k
            split_sizes = [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ]
            u = batched_mm_output[:, :, : split_sizes[0]]
            v = batched_mm_output[:, :, split_sizes[0] : split_sizes[0] + split_sizes[1]]
            q = batched_mm_output[
                :, :, split_sizes[0] + split_sizes[1] : split_sizes[0] + split_sizes[1] + split_sizes[2]
            ]
            k = batched_mm_output[:, :, split_sizes[0] + split_sizes[1] + split_sizes[2] :]
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
                print(f"          🎯 Tripy attention with relative bias ({self._normalization})")
            attn_output, cached_q_new, cached_k_new = _tripy_hstu_attention_maybe_from_cache_dense(
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
            q_heads = tp.reshape(q, (B, N, self._num_heads, self._attention_dim))
            k_heads = tp.reshape(k, (B, N, self._num_heads, self._attention_dim))
            v_heads = tp.reshape(v, (B, N, self._num_heads, self._linear_dim))

            # [B, H, N, N]
            qk_attn = tp.permute(q_heads, (0, 2, 1, 3)) @ tp.permute(  # [B, H, N, D]
                k_heads, (0, 2, 3, 1)
            )  # [B, H, D, N]
            if self._rel_attn_bias is not None:
                qk_attn = qk_attn + tp.unsqueeze(self._rel_attn_bias(all_timestamps), 1).cast(tp.float32)
            qk_attn = tp.softmax(qk_attn / math.sqrt(self._attention_dim), dim=-1)

            # Apply causal mask
            if len(invalid_attn_mask.shape) == 2:
                qk_attn = qk_attn * tp.unsqueeze(tp.unsqueeze(invalid_attn_mask, 0), 0).cast(tp.float32)
            else:
                qk_attn = qk_attn * tp.unsqueeze(invalid_attn_mask, 1).cast(tp.float32)

            v_heads_transposed = tp.permute(v_heads, (0, 2, 1, 3))  # [B, H, N, D]
            attn_result = qk_attn @ v_heads_transposed  # [B, H, N, D]
            attn_result = tp.permute(attn_result, (0, 2, 1, 3))  # [B, N, H, D]
            attn_output = tp.reshape(attn_result, (B, N, self._num_heads * self._linear_dim))

            cached_q_new, cached_k_new = q, k
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        # Combine u and attention output
        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = tp.concatenate([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        # Final linear projection with residual connection
        # Note: Padded positions remain zero due to zero inputs
        linear_output = o_input @ self._o_weight + self._o_bias
        new_outputs = linear_output + x

        # Handle caching
        cache_state = None
        if return_cache_states:
            cache_state = (v, cached_q_new, cached_k_new, new_outputs)

        return new_outputs, cache_state


class TripyHSTUDense(tp.Module):
    def __init__(
        self,
        modules: List[TripySequentialTransductionUnitDense],
        autocast_dtype: Optional[str],
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self._attention_layers: List[TripySequentialTransductionUnitDense] = modules
        self._autocast_dtype: Optional[str] = autocast_dtype
        self._verbose: bool = verbose

    def forward(
        self,
        x: tp.Tensor,  # [B, N, D] (padded with zeros beyond sequence length)
        all_timestamps: Optional[tp.Tensor],
        invalid_attn_mask: tp.Tensor,  # [B, N, N] or [N, N]
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[tp.Tensor, List[HSTUCacheState]]:
        """
        Tripy forward pass through all attention layers.
        Padded positions (zeros) are handled naturally by the operations.

        Args:
            x: [B, N, D] input embeddings (padded with zeros beyond sequence length)
            invalid_attn_mask: [B, N, N] or [N, N] causal attention mask

        Returns:
            output: [B, N, D] (padded positions remain zero)
            cache_states: List of cache states for each layer
        """
        if self._verbose:
            print(f"    🏗️  TripyHSTUDense processing {len(self._attention_layers)} layers...")

        cache_states: List[HSTUCacheState] = []

        # Note: Tripy doesn't have autocast like PyTorch, so we skip that
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


class TripyHSTUDenseModel(TripySequentialEncoderWithLearnedSimilarityModule):
    """
    Tripy implementation of HSTU that eliminates all jagged tensor operations.
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
        embedding_module: TripyEmbeddingModule,
        similarity_module: TripySimilarityModule,
        input_features_preproc_module: TripyInputFeaturesPreprocessorModule,
        output_postproc_module: TripyOutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: TripyEmbeddingModule = embedding_module
        self._input_features_preproc: TripyInputFeaturesPreprocessorModule = input_features_preproc_module
        self._output_postproc: TripyOutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias

        self._hstu = TripyHSTUDense(
            modules=[
                TripySequentialTransductionUnitDense(
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
                            bucketization_fn=lambda x: (
                                tp.log(tp.maximum(x.__abs__(), 1).cast(tp.float32)) / 0.301
                            ).cast(tp.int32),
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

        # Causal attention mask - matches PyTorch register_buffer behavior
        self._attn_mask = tp.Tensor(
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length + max_output_len,
                        self._max_sequence_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
        )
        self._verbose: bool = verbose

    def get_item_embeddings(self, item_ids: tp.Tensor) -> tp.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        debug_str = (
            f"TripyHSTUDense-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def generate_user_embeddings(
        self,
        past_lengths: tp.Tensor,
        past_ids: tp.Tensor,
        past_embeddings: tp.Tensor,
        past_payloads: Dict[str, tp.Tensor],
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[tp.Tensor, List[HSTUCacheState]]:
        """
        Tripy version of user embedding generation.
        Input embeddings are already padded with zeros beyond sequence length.

        Args:
            past_lengths: [B] sequence lengths
            past_embeddings: [B, N, D] dense embeddings (padded with zeros)

        Returns:
            user_embeddings: [B, N, D] (padded positions remain zero)
            cache_states: List of cache states
        """
        B, N, _ = past_embeddings.shape

        if self._verbose:
            print(f"  📊 Tripy input preprocessing...")
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
            print(f"  🔢 Tripy HSTU forward pass...")

        # Tripy HSTU forward pass
        # Pad to attention mask size to match behavior when converting from dense -> jagged -> dense
        attention_mask_size = self._attn_mask.shape[-1]
        current_size = user_embeddings.shape[1]

        if current_size < attention_mask_size:
            # Pad to match attention mask size, just like jagged_to_padded_dense does
            padding_size = attention_mask_size - current_size
            user_embeddings = tp.pad(
                user_embeddings, [(0, 0), (0, padding_size), (0, 0)], value=0.0
            )  # Pad sequence dimension with zeros
            if self._verbose:
                print(f"    Padded to attention mask size: {user_embeddings.shape}")
                print(f"    Padded sample [0,0,:5]: {user_embeddings[0,0,:5]}")

        user_embeddings, cached_states = self._hstu(
            x=user_embeddings,
            all_timestamps=(past_payloads[TIMESTAMPS_KEY] if TIMESTAMPS_KEY in past_payloads else None),
            invalid_attn_mask=1.0 - self._attn_mask,
            cache=cache,
            return_cache_states=return_cache_states,
        )

        if self._verbose:
            print(f"    After HSTU layers: {user_embeddings.shape}")
            print(f"    HSTU output sample [0,0,:5]: {user_embeddings[0,0,:5]}")

        # Zero out positions beyond actual sequence lengths to match jagged behavior
        # The jagged implementation naturally excludes these positions from computation
        for i in range(B):
            length = past_lengths[i]
            seq_len = user_embeddings.shape[1]
            if self._verbose:
                print(f"    Tripy: Batch {i} length={length}, seq_len={seq_len}")
            if length < seq_len.cast(tp.int64):
                if self._verbose:
                    print(f"    Tripy: Zeroing batch {i} positions {length}:{seq_len} (length={length})")
                # Create a mask for positions beyond the sequence length
                mask = tp.arange(seq_len).cast(tp.int64) >= length
                mask_expanded = tp.unsqueeze(mask, -1)  # [N, 1]
                # Apply mask only to the current batch
                batch_embeddings = user_embeddings[i]  # [N, D]
                batch_embeddings = tp.where(mask_expanded, 0.0, batch_embeddings)
                # Update the tensor (Tripy doesn't support item assignment, so we reconstruct)
                if i == 0:
                    user_embeddings = tp.concatenate([tp.unsqueeze(batch_embeddings, 0), user_embeddings[1:]], dim=0)
                elif i == B - 1:
                    user_embeddings = tp.concatenate([user_embeddings[:i], tp.unsqueeze(batch_embeddings, 0)], dim=0)
                else:
                    user_embeddings = tp.concatenate(
                        [user_embeddings[:i], tp.unsqueeze(batch_embeddings, 0), user_embeddings[i + 1 :]], dim=0
                    )

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
        past_lengths: tp.Tensor,
        past_ids: tp.Tensor,
        past_embeddings: tp.Tensor,
        past_payloads: Dict[str, tp.Tensor],
        batch_id: Optional[int] = None,
    ) -> tp.Tensor:
        """Tripy forward pass."""
        if self._verbose:
            print(f"🚀 TripyHSTUDense.forward() → generate_user_embeddings()")

        encoded_embeddings, _ = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        if self._verbose:
            print(f"✅ TripyHSTUDense.forward() complete: {encoded_embeddings.shape}")

        return encoded_embeddings

    def _encode(
        self,
        past_lengths: tp.Tensor,
        past_ids: tp.Tensor,
        past_embeddings: tp.Tensor,
        past_payloads: Dict[str, tp.Tensor],
        cache: Optional[List[HSTUCacheState]],
        return_cache_states: bool,
    ) -> Union[tp.Tensor, Tuple[tp.Tensor, List[HSTUCacheState]]]:
        """Tripy encoding to get current embeddings."""
        encoded_seq_embeddings, cache_states = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        current_embeddings = tripy_get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )
        if return_cache_states:
            return current_embeddings, cache_states
        else:
            return current_embeddings

    def encode(
        self,
        past_lengths: tp.Tensor,
        past_ids: tp.Tensor,
        past_embeddings: tp.Tensor,
        past_payloads: Dict[str, tp.Tensor],
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Union[tp.Tensor, Tuple[tp.Tensor, List[HSTUCacheState]]]:
        """Tripy encoding interface."""
        if self._verbose:
            print(f"🎯 TripyHSTUDense.encode() → extracting current embeddings")

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
                print(f"✅ TripyHSTUDense.encode() complete: {result[0].shape}")
            else:
                print(f"✅ TripyHSTUDense.encode() complete: {result.shape}")

        return result
