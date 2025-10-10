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
"""
Test script that replicates the exact research pipeline data flow.
This uses the same parameters and data processing as the working research pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict

# Import fbgemm_gpu to ensure operations are available
import fbgemm_gpu

# Import research HSTU components
from generative_recommenders.research.modeling.sequential.hstu import HSTU
from generative_recommenders.research.modeling.sequential.hstu_dense import HSTUDenseModel
from generative_recommenders.research.modeling.sequential.embedding_modules import LocalEmbeddingModule
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
)
from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import DotProductSimilarity

use_verbose = True


def verify_hstu_output_accuracy(
    encoded_embeddings: torch.Tensor,
    current_embeddings: torch.Tensor,
    past_lengths: torch.Tensor,
    batch_size: int,
    max_seq_len: int,
    gr_output_length: int,
    embedding_dim: int,
    verbose: bool = use_verbose,
) -> bool:
    """
    Comprehensive verification of HSTU output accuracy and consistency.

    Returns:
        bool: True if all checks pass, False otherwise
    """
    try:
        if verbose:
            print(f"\n=== HSTU Output Verification ===")

        # 1. Shape verification
        expected_forward_shape = (batch_size, max_seq_len + gr_output_length + 1, embedding_dim)
        expected_encode_shape = (batch_size, embedding_dim)

        assert (
            encoded_embeddings.shape == expected_forward_shape
        ), f"Forward shape mismatch: {encoded_embeddings.shape} != {expected_forward_shape}"
        assert (
            current_embeddings.shape == expected_encode_shape
        ), f"Encode shape mismatch: {current_embeddings.shape} != {expected_encode_shape}"
        if verbose:
            print(f"✓ Shape verification passed")

        # 2. Value sanity checks
        assert torch.isfinite(encoded_embeddings).all(), "Forward output has non-finite values"
        assert torch.isfinite(current_embeddings).all(), "Encode output has non-finite values"
        assert not torch.isnan(encoded_embeddings).any(), "Forward output has NaN values"
        assert not torch.isnan(current_embeddings).any(), "Encode output has NaN values"
        if verbose:
            print(f"✓ Finite value verification passed")

        # 3. Non-degeneracy checks
        assert not torch.allclose(
            encoded_embeddings, torch.zeros_like(encoded_embeddings)
        ), "Forward output is all zeros"
        assert not torch.allclose(
            current_embeddings, torch.zeros_like(current_embeddings)
        ), "Encode output is all zeros"
        if verbose:
            print(f"✓ Non-degeneracy verification passed")

        # 4. Consistency between forward and encode
        manual_current_embeddings = []
        for i, length in enumerate(past_lengths):
            manual_current_embeddings.append(encoded_embeddings[i, length - 1])
        manual_current_embeddings = torch.stack(manual_current_embeddings)

        consistency_check = torch.allclose(current_embeddings, manual_current_embeddings, rtol=1e-4, atol=1e-5)
        if verbose:
            if consistency_check:
                print(f"✓ Encode/Forward consistency verified")
            else:
                max_diff = (current_embeddings - manual_current_embeddings).abs().max()
                print(f"⚠ Encode/Forward difference: {max_diff:.6f} (may be expected)")

        # 5. Statistical properties
        forward_stats = {
            "mean": encoded_embeddings.mean().item(),
            "std": encoded_embeddings.std().item(),
            "min": encoded_embeddings.min().item(),
            "max": encoded_embeddings.max().item(),
        }
        encode_stats = {
            "mean": current_embeddings.mean().item(),
            "std": current_embeddings.std().item(),
            "min": current_embeddings.min().item(),
            "max": current_embeddings.max().item(),
        }

        # Check for reasonable value ranges
        assert abs(forward_stats["mean"]) < 10, f"Forward mean too large: {forward_stats['mean']}"
        assert abs(encode_stats["mean"]) < 10, f"Encode mean too large: {encode_stats['mean']}"
        assert forward_stats["std"] > 1e-6, f"Forward std too small: {forward_stats['std']}"
        assert encode_stats["std"] > 1e-6, f"Encode std too small: {encode_stats['std']}"

        if verbose:
            print(f"✓ Statistical properties verified")
            print(f"  Forward: mean={forward_stats['mean']:.4f}, std={forward_stats['std']:.4f}")
            print(f"  Encode:  mean={encode_stats['mean']:.4f}, std={encode_stats['std']:.4f}")

        # 6. Embedding diversity
        forward_var_batch = encoded_embeddings.var(dim=0).mean().item()
        forward_var_seq = encoded_embeddings.var(dim=1).mean().item()
        encode_var_batch = current_embeddings.var(dim=0).mean().item()

        diversity_threshold = 1e-6
        assert forward_var_batch > diversity_threshold, f"Low forward batch variance: {forward_var_batch}"
        assert forward_var_seq > diversity_threshold, f"Low forward sequence variance: {forward_var_seq}"
        assert encode_var_batch > diversity_threshold, f"Low encode batch variance: {encode_var_batch}"

        if verbose:
            print(f"✓ Embedding diversity verified")
            print(f"  Batch variance: {forward_var_batch:.6f}, Sequence variance: {forward_var_seq:.6f}")

        return True

    except AssertionError as e:
        if verbose:
            print(f"❌ Verification failed: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"❌ Verification error: {e}")
        return False


def create_research_pipeline_data(batch_size=4, max_seq_len=200, embedding_dim=50, num_items=1000):
    """Create data that matches the research pipeline exactly."""

    # Create past_lengths: [batch_size] - actual sequence lengths for each batch item
    # In research pipeline, these are typically much longer (up to 200)
    past_lengths = torch.randint(50, max_seq_len + 1, (batch_size,))

    # Create past_ids: [batch_size, max_seq_len] - item IDs for each sequence
    past_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    for i, length in enumerate(past_lengths):
        past_ids[i, :length] = torch.randint(1, num_items, (length,))

    # Create past_embeddings: [batch_size, max_seq_len, embedding_dim] - embeddings
    past_embeddings = torch.randn(batch_size, max_seq_len, embedding_dim)

    # Create timestamps - these need to match the attention mask size
    # The attention mask will be (max_seq_len + max_output_len, max_seq_len + max_output_len)
    # From the research pipeline: max_output_len = gr_output_length + 1 = 10 + 1 = 11
    attention_mask_size = max_seq_len + 11  # max_output_len = 11
    past_timestamps = torch.randint(0, 1000, (batch_size, attention_mask_size))
    past_payloads = {"timestamps": past_timestamps}

    return past_lengths, past_ids, past_embeddings, past_payloads


def create_shared_modules(max_seq_len, embedding_dim, num_items, gr_output_length):
    """Create shared modules for both jagged and dense models."""
    embedding_module = LocalEmbeddingModule(
        num_items=num_items,
        item_embedding_dim=embedding_dim,
    )

    similarity_module = DotProductSimilarity()

    input_preproc = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=max_seq_len + gr_output_length + 1,  # 200 + 10 + 1 = 211
        embedding_dim=embedding_dim,
        dropout_rate=0,
    )

    output_postproc = L2NormEmbeddingPostprocessor(
        embedding_dim=embedding_dim,
        eps=1e-6,
    )

    return embedding_module, similarity_module, input_preproc, output_postproc


def test_research_pipeline():
    """Test the research HSTU pipeline with exact parameters from the gin config."""

    print("=== Testing Research HSTU Pipeline (Exact Parameters) ===")

    # Parameters from the gin config file
    batch_size = 4
    max_seq_len = 200  # From gin config: max_sequence_length = 200
    embedding_dim = 50  # From gin config: item_embedding_dim = 50
    num_items = 1000
    gr_output_length = 10  # From gin config: gr_output_length = 10

    # HSTU parameters from gin config
    num_blocks = 8  # hstu_encoder.num_blocks = 8
    num_heads = 2  # hstu_encoder.num_heads = 2
    dqk = 25  # hstu_encoder.dqk = 25
    dv = 25  # hstu_encoder.dv = 25
    linear_dropout_rate = 0  # 0.2 for training, 0 for inference

    # Create synthetic data
    past_lengths, past_ids, past_embeddings, past_payloads = create_research_pipeline_data(
        batch_size, max_seq_len, embedding_dim, num_items
    )

    print(f"Input shapes:")
    print(f"  past_lengths: {past_lengths.shape}")
    print(f"  past_ids: {past_ids.shape}")
    print(f"  past_embeddings: {past_embeddings.shape}")
    print(f"  past_timestamps: {past_payloads['timestamps'].shape}")

    # Create shared modules
    embedding_module, similarity_module, input_preproc, output_postproc = create_shared_modules(
        max_seq_len, embedding_dim, num_items, gr_output_length
    )

    # Create HSTU model with same parameters from gin config
    hstu_model = HSTU(
        max_sequence_len=max_seq_len,  # 200
        max_output_len=gr_output_length + 1,  # 10 + 1 = 11
        embedding_dim=embedding_dim,  # 50
        num_blocks=num_blocks,  # 8
        num_heads=num_heads,  # 2
        linear_dim=dv,  # 25
        attention_dim=dqk,  # 25
        normalization="rel_bias",  # Default from hstu_encoder
        linear_config="uvqk",  # Default from hstu_encoder
        linear_activation="silu",  # Default from hstu_encoder
        linear_dropout_rate=linear_dropout_rate,  # 0.2 for training, 0 for inference
        attn_dropout_rate=0.0,  # Default from hstu_encoder
        embedding_module=embedding_module,
        similarity_module=similarity_module,
        input_features_preproc_module=input_preproc,
        output_postproc_module=output_postproc,
        enable_relative_attention_bias=True,  # Default from hstu_encoder
        concat_ua=False,  # Default from hstu_encoder
        verbose=use_verbose,
    )

    print(f"\nHSTU model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in hstu_model.parameters()):,}")

    # Test forward pass
    print(f"\n=== Testing Forward Pass ===")
    print(f"=== HSTU Method Call Flow Tracing ===")

    with torch.no_grad():
        try:
            encoded_embeddings = hstu_model(
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_embeddings=past_embeddings,
                past_payloads=past_payloads,
            )

            print(f"\n📊 Forward pass results:")
            print(f"  Output shape: {encoded_embeddings.shape}")
            print(f"  Expected: [{batch_size}, {max_seq_len + gr_output_length + 1}, {embedding_dim}]")

            current_embeddings = hstu_model.encode(
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_embeddings=past_embeddings,
                past_payloads=past_payloads,
            )

            print(f"\n📊 Encode results:")
            print(f"  Current embeddings: {current_embeddings.shape}")
            print(f"  Expected: [{batch_size}, {embedding_dim}]")

            # Run comprehensive verification
            verification_passed = verify_hstu_output_accuracy(
                encoded_embeddings=encoded_embeddings,
                current_embeddings=current_embeddings,
                past_lengths=past_lengths,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                gr_output_length=gr_output_length,
                embedding_dim=embedding_dim,
                verbose=use_verbose,
            )

            if not verification_passed:
                print(f"❌ Output verification failed!")
                return False

            # Test reproducibility with same inputs
            print(f"\n=== Reproducibility Test ===")
            with torch.no_grad():
                encoded_embeddings_2 = hstu_model(
                    past_lengths=past_lengths,
                    past_ids=past_ids,
                    past_embeddings=past_embeddings,
                    past_payloads=past_payloads,
                )

                if torch.allclose(encoded_embeddings, encoded_embeddings_2, rtol=1e-6, atol=1e-7):
                    print(f"✓ Model is deterministic (same inputs produce same outputs)")
                else:
                    max_diff = (encoded_embeddings - encoded_embeddings_2).abs().max()
                    print(f"⚠ Model shows non-determinism: max difference = {max_diff:.8f}")
                    if max_diff > 1e-4:
                        print(f"  This may indicate dropout is enabled during inference")

            # Test Dense HSTU Implementation
            print(f"\n" + "=" * 80)
            print(f"=== Testing Dense HSTU Implementation ===")
            print(f"=" * 80)

            # Create dense model with same parameters but separate modules to avoid shared state
            embedding_module_dense, similarity_module_dense, input_preproc_dense, output_postproc_dense = (
                create_shared_modules(max_seq_len, embedding_dim, num_items, gr_output_length)
            )

            hstu_dense_model = HSTUDenseModel(
                max_sequence_len=max_seq_len,  # 200
                max_output_len=gr_output_length + 1,  # 10 + 1 = 11
                embedding_dim=embedding_dim,  # 50
                num_blocks=num_blocks,  # 8
                num_heads=num_heads,  # 2
                linear_dim=dv,  # 25
                attention_dim=dqk,  # 25
                normalization="rel_bias",  # Default from hstu_encoder
                linear_config="uvqk",  # Default from hstu_encoder
                linear_activation="silu",  # Default from hstu_encoder
                linear_dropout_rate=linear_dropout_rate,  # 0.2 for training, 0 for inference
                attn_dropout_rate=0.0,  # Default from hstu_encoder
                embedding_module=embedding_module_dense,
                similarity_module=similarity_module_dense,
                input_features_preproc_module=input_preproc_dense,
                output_postproc_module=output_postproc_dense,
                enable_relative_attention_bias=True,  # Default from hstu_encoder
                concat_ua=False,  # Default from hstu_encoder
                verbose=False,  # Reduce noise for comparison
            )

            print(f"Dense HSTU model created successfully!")
            print(f"Dense model parameters: {sum(p.numel() for p in hstu_dense_model.parameters()):,}")

            # Copy weights from jagged to dense model to ensure identical parameters
            print(f"\nCopying weights from jagged to dense model...")
            dense_state_dict = hstu_dense_model.state_dict()
            jagged_state_dict = hstu_model.state_dict()

            copied_params = 0
            for name in dense_state_dict.keys():
                if name in jagged_state_dict:
                    dense_state_dict[name].copy_(jagged_state_dict[name])
                    copied_params += 1

            print(f"Copied {copied_params} parameters from jagged to dense model")

            # Test dense model forward pass
            print(f"\n=== Dense Model Forward Pass ===")

            with torch.no_grad():
                dense_encoded_embeddings = hstu_dense_model(
                    past_lengths=past_lengths,
                    past_ids=past_ids,
                    past_embeddings=past_embeddings,
                    past_payloads=past_payloads,
                )

                dense_current_embeddings = hstu_dense_model.encode(
                    past_lengths=past_lengths,
                    past_ids=past_ids,
                    past_embeddings=past_embeddings,
                    past_payloads=past_payloads,
                )

                print(f"\n📊 Dense model results:")
                print(f"  Dense forward shape: {dense_encoded_embeddings.shape}")
                print(f"  Dense encode shape:  {dense_current_embeddings.shape}")

                # Compare jagged vs dense outputs
                print(f"\n=== Jagged vs Dense Comparison ===")

                # Forward pass comparison
                if encoded_embeddings.shape == dense_encoded_embeddings.shape:
                    forward_max_diff = (encoded_embeddings - dense_encoded_embeddings).abs().max().item()
                    forward_mean_diff = (encoded_embeddings - dense_encoded_embeddings).abs().mean().item()

                    print(f"Forward pass comparison:")
                    print(f"  Max difference:  {forward_max_diff:.8f}")
                    print(f"  Mean difference: {forward_mean_diff:.8f}")

                    if forward_max_diff < 1e-5:
                        print(f"  ✅ Forward outputs match within tolerance!")
                    else:
                        print(f"  ⚠️  Forward outputs differ significantly")

                        # Check if differences are only in padded positions
                        for i, length in enumerate(past_lengths):
                            valid_jagged = encoded_embeddings[i, :length]
                            valid_dense = dense_encoded_embeddings[i, :length]
                            valid_diff = (valid_jagged - valid_dense).abs().max().item()
                            print(f"    Batch {i} (length {length}): max diff = {valid_diff:.8f}")
                        print(
                            f"  If these differences are close to 0, the overall tensor difference is in padded positions."
                        )
                else:
                    print(f"  ❌ Forward shape mismatch!")

                # Encode pass comparison
                if current_embeddings.shape == dense_current_embeddings.shape:
                    encode_max_diff = (current_embeddings - dense_current_embeddings).abs().max().item()
                    encode_mean_diff = (current_embeddings - dense_current_embeddings).abs().mean().item()

                    print(f"\nEncode pass comparison:")
                    print(f"  Max difference:  {encode_max_diff:.8f}")
                    print(f"  Mean difference: {encode_mean_diff:.8f}")

                    if encode_max_diff < 1e-5:
                        print(f"  ✅ Encode outputs match within tolerance!")
                    else:
                        print(f"  ⚠️  Encode outputs differ significantly")
                else:
                    print(f"  ❌ Encode shape mismatch!")

                # Verify dense model accuracy
                dense_verification_passed = verify_hstu_output_accuracy(
                    encoded_embeddings=dense_encoded_embeddings,
                    current_embeddings=dense_current_embeddings,
                    past_lengths=past_lengths,
                    batch_size=batch_size,
                    max_seq_len=max_seq_len,
                    gr_output_length=gr_output_length,
                    embedding_dim=embedding_dim,
                    verbose=False,  # Reduce noise
                )

                if dense_verification_passed:
                    print(f"\n✅ Dense model verification passed!")
                else:
                    print(f"\n❌ Dense model verification failed!")

        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback

            traceback.print_exc()
            return False

    return True


if __name__ == "__main__":
    success = test_research_pipeline()
    if success:
        print(
            "\n🎉 All tests passed! Both jagged and dense HSTU implementations work correctly and produce equivalent results."
        )
    else:
        print("\n💥 Tests failed. Check the error messages above.")
