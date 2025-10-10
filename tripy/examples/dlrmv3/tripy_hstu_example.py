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
Tripy HSTU Example: Compare PyTorch Dense vs Tripy implementations with compilation.
"""

import os
import time
import argparse
from typing import Dict, Tuple

import torch
import numpy as np
import nvtripy as tp

# Import PyTorch dense implementation
from generative_recommenders.research.modeling.sequential.hstu_dense import HSTUDenseModel
from generative_recommenders.research.modeling.sequential.embedding_modules import LocalEmbeddingModule
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
)
from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import DotProductSimilarity

# Import Tripy implementation
from generative_recommenders.research.modeling.sequential.tripy_hstu_dense import TripyHSTUDenseModel
from generative_recommenders.research.modeling.sequential.tripy_embedding_modules import TripyLocalEmbeddingModule
from generative_recommenders.research.modeling.sequential.tripy_input_features_preprocessors import (
    TripyLearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.tripy_output_postprocessors import (
    TripyL2NormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.tripy_similarity_module import TripyDotProductSimilarity


def create_test_data(batch_size: int, max_seq_len: int, embedding_dim: int, num_items: int) -> Tuple:
    """Create test data for HSTU comparison."""

    # Create past_lengths: [batch_size] - actual sequence lengths
    past_lengths = torch.randint(50, max_seq_len + 1, (batch_size,))

    # Create past_ids: [batch_size, max_seq_len] - item IDs
    past_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    for i, length in enumerate(past_lengths):
        past_ids[i, :length] = torch.randint(1, num_items, (length,))

    # Create past_embeddings: [batch_size, max_seq_len, embedding_dim] - padded with zeros
    past_embeddings = torch.zeros(batch_size, max_seq_len, embedding_dim)
    for i, length in enumerate(past_lengths):
        past_embeddings[i, :length] = torch.randn(length, embedding_dim)

    # Create timestamps
    attention_mask_size = max_seq_len + 11  # max_output_len = 11
    past_timestamps = torch.randint(0, 1000, (batch_size, attention_mask_size))
    past_payloads = {"timestamps": past_timestamps}

    return past_lengths, past_ids, past_embeddings, past_payloads


def create_pytorch_model(max_seq_len: int, embedding_dim: int, num_items: int, gr_output_length: int) -> HSTUDenseModel:
    """Create PyTorch dense HSTU model matching research pipeline."""

    # Model parameters from research pipeline gin config
    num_blocks = 8  # hstu_encoder.num_blocks = 8
    num_heads = 2  # hstu_encoder.num_heads = 2
    dqk = 25  # hstu_encoder.dqk = 25
    dv = 25  # hstu_encoder.dv = 25

    # Create modules
    embedding_module = LocalEmbeddingModule(
        num_items=num_items,
        item_embedding_dim=embedding_dim,
    )

    similarity_module = DotProductSimilarity()

    input_preproc = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=max_seq_len + gr_output_length + 1,
        embedding_dim=embedding_dim,
        dropout_rate=0,
    )

    output_postproc = L2NormEmbeddingPostprocessor(
        embedding_dim=embedding_dim,
        eps=1e-6,
    )

    # Create model
    model = HSTUDenseModel(
        max_sequence_len=max_seq_len,
        max_output_len=gr_output_length + 1,
        embedding_dim=embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        linear_dim=dv,
        attention_dim=dqk,
        normalization="rel_bias",
        linear_config="uvqk",
        linear_activation="silu",
        linear_dropout_rate=0,
        attn_dropout_rate=0,
        embedding_module=embedding_module,
        similarity_module=similarity_module,
        input_features_preproc_module=input_preproc,
        output_postproc_module=output_postproc,
        enable_relative_attention_bias=True,
        concat_ua=False,
        verbose=False,
    )

    return model


def create_tripy_model(
    max_seq_len: int, embedding_dim: int, num_items: int, gr_output_length: int
) -> TripyHSTUDenseModel:
    """Create Tripy HSTU model matching research pipeline."""

    # Model parameters from research pipeline gin config (same as PyTorch)
    num_blocks = 8  # hstu_encoder.num_blocks = 8
    num_heads = 2  # hstu_encoder.num_heads = 2
    dqk = 25  # hstu_encoder.dqk = 25
    dv = 25  # hstu_encoder.dv = 25

    # Create Tripy modules
    embedding_module = TripyLocalEmbeddingModule(
        num_items=num_items,
        item_embedding_dim=embedding_dim,
    )

    similarity_module = TripyDotProductSimilarity()

    input_preproc = TripyLearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=max_seq_len + gr_output_length + 1,
        embedding_dim=embedding_dim,
        dropout_rate=0,
    )

    output_postproc = TripyL2NormEmbeddingPostprocessor(
        embedding_dim=embedding_dim,
        eps=1e-6,
    )

    # Create model
    model = TripyHSTUDenseModel(
        max_sequence_len=max_seq_len,
        max_output_len=gr_output_length + 1,
        embedding_dim=embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        linear_dim=dv,
        attention_dim=dqk,
        normalization="rel_bias",
        linear_config="uvqk",
        linear_activation="silu",
        linear_dropout_rate=0,
        attn_dropout_rate=0,
        embedding_module=embedding_module,
        similarity_module=similarity_module,
        input_features_preproc_module=input_preproc,
        output_postproc_module=output_postproc,
        enable_relative_attention_bias=True,
        concat_ua=False,
        verbose=False,
    )

    return model


def copy_weights_to_tripy(pytorch_model: HSTUDenseModel, tripy_model: TripyHSTUDenseModel):
    """Copy weights from PyTorch model to Tripy model."""

    pytorch_state = pytorch_model.state_dict()
    tripy_state = tripy_model.state_dict()

    print(f"PyTorch model has {len(pytorch_state)} parameters")
    print(f"Tripy model has {len(tripy_state)} parameters")

    # Copy matching parameters
    copied_count = 0
    missing_params = []

    for name, tripy_param in tripy_state.items():
        if name in pytorch_state:
            # Convert PyTorch tensor to Tripy tensor
            pytorch_param = pytorch_state[name]
            tripy_state[name] = tp.Tensor(pytorch_param.detach().cpu())
            copied_count += 1
        else:
            missing_params.append(name)

    # Load the updated state dict
    tripy_model.load_state_dict(tripy_state)

    print(f"Copied {copied_count} parameters from PyTorch to Tripy model")
    if missing_params:
        print(f"Missing parameters in PyTorch model: {missing_params[:5]}...")  # Show first 5

    # Check for any remaining uninitialized parameters and initialize them
    final_state = tripy_model.state_dict()
    uninitialized = [name for name, param in final_state.items() if not isinstance(param, tp.Tensor)]
    if uninitialized:
        print(f"Initializing remaining {len(uninitialized)} parameters with dummy values...")
        tripy_model.initialize_dummy_parameters()
        print(f"All parameters now initialized!")


def compile_tripy_model(
    model: TripyHSTUDenseModel, sample_inputs: Tuple, engine_dir: str, verbose: bool = False
) -> tp.Executable:
    """Compile Tripy model with caching."""

    os.makedirs(engine_dir, exist_ok=True)
    engine_path = os.path.join(engine_dir, "tripy_hstu_dense.engine")

    # Check if cached engine exists
    if os.path.exists(engine_path):
        if verbose:
            print(f"Loading cached engine from {engine_path}")
        return tp.Executable.load(engine_path)

    if verbose:
        print("Compiling TripyHSTUDenseModel...", end=" ", flush=True)
        compile_start = time.perf_counter()

    # Create InputInfo objects from sample inputs
    past_lengths, past_ids, past_embeddings, past_payloads = sample_inputs
    # print(past_payloads)
    tripy_inputs = [
        tp.InputInfo(shape=past_lengths.shape, dtype=tp.int32),  # past_lengths
        tp.InputInfo(shape=past_ids.shape, dtype=tp.int32),  # past_ids
        tp.InputInfo(shape=past_embeddings.shape, dtype=tp.float32),  # past_embeddings
        {
            k: tp.InputInfo(shape=v.shape, dtype=tp.int64 if k == "timestamps" else tp.float32)
            for k, v in past_payloads.items()
        },  # past_payloads
    ]

    compiled_model = tp.compile(model, args=tripy_inputs, optimization_level=5)
    compiled_model.save(engine_path)

    if verbose:
        compile_end = time.perf_counter()
        print(f"saved engine to {engine_path} ({compile_end - compile_start:.2f}s)")

    return compiled_model


def compare_outputs(
    pytorch_output: torch.Tensor, tripy_output: tp.Tensor, tolerance: float = 2e-2, detailed: bool = False
) -> bool:
    """Compare PyTorch and Tripy outputs."""

    tripy_torch = torch.from_dlpack(tripy_output).to("cpu")

    # Compare shapes
    if pytorch_output.shape != tripy_torch.shape:
        print(f"❌ Shape mismatch: PyTorch {pytorch_output.shape} vs Tripy {tripy_torch.shape}")
        return False

    # Compare values
    diff = (pytorch_output - tripy_torch).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Output comparison:")
    print(f"  Max difference:  {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")

    if detailed and max_diff >= tolerance:
        print(f"\n=== Detailed Analysis ===")

        # Find location of max difference
        max_idx = diff.argmax()
        max_coords = torch.unravel_index(max_idx, diff.shape)
        print(f"Max difference location: {max_coords}")
        print(
            f"PyTorch value: {pytorch_output[max_coords[0], max_coords[1].item():max_coords[1].item()+5, max_coords[2].item()]}"
        )
        print(
            f"Tripy value: {tripy_torch[max_coords[0], max_coords[1].item():max_coords[1].item()+5, max_coords[2].item()]}"
        )

        # Statistics per batch
        if len(diff.shape) >= 2:
            for b in range(diff.shape[0]):
                batch_max = diff[b].max().item()
                batch_mean = diff[b].mean().item()
                print(f"Batch {b}: max_diff={batch_max:.8f}, mean_diff={batch_mean:.8f}")

        # Show first few differences
        print(f"\nFirst 5x5 differences:")
        if len(diff.shape) == 2:
            print(diff[:5, :5])
        elif len(diff.shape) == 3:
            print(diff[0, :5, :5])

    if max_diff < tolerance:
        print(f"  ✅ Outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"  ⚠️  Outputs differ significantly")
        return False


def benchmark_models(
    pytorch_model: HSTUDenseModel, compiled_tripy_model: tp.Executable, test_inputs: Tuple, num_runs: int = 10
) -> Tuple[float, float]:
    """Benchmark PyTorch vs compiled Tripy models."""

    past_lengths, past_ids, past_embeddings, past_payloads = test_inputs

    # Benchmark PyTorch model
    pytorch_times = []
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = pytorch_model(past_lengths, past_ids, past_embeddings, past_payloads)

        # Benchmark
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = pytorch_model(past_lengths, past_ids, past_embeddings, past_payloads)
            pytorch_times.append(time.perf_counter() - start_time)

    # Benchmark Tripy model
    tripy_inputs = [
        tp.Tensor(past_lengths.numpy()),
        tp.Tensor(past_ids.numpy()),
        tp.Tensor(past_embeddings.numpy()),
        {k: tp.Tensor(v.numpy()) for k, v in past_payloads.items()},
    ]

    tripy_times = []
    # Warmup
    for _ in range(3):
        _ = compiled_tripy_model(*tripy_inputs)

    # Benchmark
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = compiled_tripy_model(*tripy_inputs)
        tripy_times.append(time.perf_counter() - start_time)

    pytorch_avg = np.mean(pytorch_times) * 1000  # Convert to ms
    tripy_avg = np.mean(tripy_times) * 1000

    return pytorch_avg, tripy_avg


def main():
    parser = argparse.ArgumentParser(description="Tripy HSTU Example")
    # Research pipeline parameters from gin config
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-seq-len", type=int, default=200, help="Maximum sequence length (from gin config)")
    parser.add_argument("--embedding-dim", type=int, default=50, help="Embedding dimension (from gin config)")
    parser.add_argument("--num-items", type=int, default=1000, help="Number of items")
    parser.add_argument("--engine-dir", type=str, default="engines", help="Engine cache directory")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    print("=== Tripy HSTU Dense Implementation Test (Research Pipeline Parameters) ===")
    print(f"Batch size: {args.batch_size}, Max seq len: {args.max_seq_len} (gin config)")
    print(f"Embedding dim: {args.embedding_dim} (gin config), Num items: {args.num_items}")
    print(f"Model: 8 blocks, 2 heads, dqk=25, dv=25 (gin config)")

    # Create test data
    gr_output_length = 10
    test_inputs = create_test_data(args.batch_size, args.max_seq_len, args.embedding_dim, args.num_items)
    past_lengths, past_ids, past_embeddings, past_payloads = test_inputs

    print(f"\nTest data shapes:")
    print(f"  past_lengths: {past_lengths.shape}")
    print(f"  past_ids: {past_ids.shape}")
    print(f"  past_embeddings: {past_embeddings.shape}")
    print(f"  past_timestamps: {past_payloads['timestamps'].shape}")

    # Create models
    print(f"\nCreating models...")
    pytorch_model = create_pytorch_model(args.max_seq_len, args.embedding_dim, args.num_items, gr_output_length)
    tripy_model = create_tripy_model(args.max_seq_len, args.embedding_dim, args.num_items, gr_output_length)

    print(f"PyTorch model parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")

    # Copy weights
    print(f"\nCopying weights from PyTorch to Tripy model...")
    copy_weights_to_tripy(pytorch_model, tripy_model)

    # Debug: Print sequence lengths first
    print(f"\nSequence lengths: {past_lengths}")

    # Test PyTorch model with debug
    print(f"\nTesting PyTorch model...")
    pytorch_model._verbose = True  # Enable verbose mode for debugging
    with torch.no_grad():
        pytorch_output = pytorch_model(past_lengths, past_ids, past_embeddings, past_payloads)
    print(f"PyTorch output shape: {pytorch_output.shape}")

    # Compile Tripy model
    print(f"\nCompiling Tripy model...")
    compiled_tripy_model = compile_tripy_model(tripy_model, test_inputs, args.engine_dir, args.verbose)

    # Test compiled Tripy model with debug
    print(f"\nTesting compiled Tripy model...")
    tripy_inputs = [
        tp.Tensor(past_lengths.numpy()),
        tp.Tensor(past_ids.numpy()),
        tp.Tensor(past_embeddings.numpy()),
        {k: tp.Tensor(v.numpy()) for k, v in past_payloads.items()},
    ]
    tripy_output = compiled_tripy_model(*tripy_inputs)
    print(f"Tripy output shape: {tripy_output.shape}")

    # Compare outputs
    print(f"\n=== Output Comparison ===")
    outputs_match = compare_outputs(pytorch_output, tripy_output, detailed=True)

    # Benchmark if requested
    if args.benchmark:
        print(f"\n=== Performance Benchmark ===")
        pytorch_time, tripy_time = benchmark_models(pytorch_model, compiled_tripy_model, test_inputs)

        print(f"PyTorch Dense:    {pytorch_time:.2f} ms")
        print(f"Tripy Compiled:   {tripy_time:.2f} ms")
        print(f"Speedup:          {pytorch_time / tripy_time:.2f}x")

    if args.benchmark:
        print(f"\n=== Benchmark ===")
        print(f"🚀 Tripy compiled model is {pytorch_time / tripy_time:.2f}x faster")

    print("Test completed!")


if __name__ == "__main__":
    main()
