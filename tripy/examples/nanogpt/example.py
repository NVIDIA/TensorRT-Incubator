#
# SPDX-FileCopyrightText: Copyright (c) 2024-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

import argparse
import time

import tiktoken
import torch
from model import GPT, GPTConfig
from weight_loader import load_quant_weights_from_hf, load_weights_from_hf

import nvtripy as tp


def initialize_gpt_model(model_type, padded_seq_len, dtype, quant_mode):
    # num_layers, num_heads and embedding_size are determined from model_type while other
    # parameters are always the same.
    num_layers, num_heads, embedding_size = {
        "gpt2": (12, 12, 768),  # 124M params
        "gpt2-medium": (24, 16, 1024),  # 350M params
        "gpt2-large": (36, 20, 1280),  # 774M params
        "gpt2-xl": (48, 25, 1600),  # 1558M params
    }[model_type]

    config = GPTConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_size=embedding_size,
        vocab_size=50257,
        block_size=1024,
        bias=True,
        seq_len=padded_seq_len,
        batch_size=1,
        dtype=dtype,
        quant_mode=quant_mode,
    )
    model = GPT(config)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Runs NanoGPT inference with the given input text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-text", type=str, help="The input text", required=True)
    parser.add_argument(
        "--model-type",
        type=str,
        help="The GPT variant to use.",
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    )
    parser.add_argument("--max-new-tokens", type=int, help="The maximum number of new tokens to generate", default=10)
    parser.add_argument("--seed", type=int, help="The seed to use for psuedo-random number generation.", default=None)
    parser.add_argument(
        "--quant-mode",
        type=str,
        help="Quantization mode.",
        choices=["int8-weight-only", "int4-weight-only", "float8"],
    )

    args = parser.parse_args()

    encoder = tiktoken.get_encoding("gpt2")

    input_ids = encoder.encode(args.input_text, allowed_special={"<|endoftext|>"})

    TEMPERATURE = 0.8
    TOP_K = 5
    MODEL_DTYPE = tp.float16

    padded_seq_len = len(input_ids) + args.max_new_tokens

    model = initialize_gpt_model(args.model_type, padded_seq_len, MODEL_DTYPE, args.quant_mode)
    if not args.quant_mode:
        load_weights_from_hf(model, args.model_type, MODEL_DTYPE)
    else:
        load_quant_weights_from_hf(model, args.model_type, MODEL_DTYPE, args.quant_mode)

    idx = tp.reshape(tp.Tensor(input_ids), shape=(1, len(input_ids)))

    # Compile the model before running inference.
    print(f"Compiling the model (this may take a few seconds)...")
    compile_start_time = time.perf_counter()
    input_shape = (
        1,
        # We can specify dynamic dimensions by using a sequence indicating the min/opt/max values that
        # a dimension should support:
        (1, len(input_ids), padded_seq_len),
    )
    model = tp.compile(model, args=[tp.InputInfo(input_shape, dtype=tp.int32)])
    compile_end_time = time.perf_counter()
    print(f"Compilation took {compile_end_time - compile_start_time} seconds.")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed)

    start_time = time.perf_counter()
    for token_idx in range(len(input_ids), len(input_ids) + args.max_new_tokens):
        logits = model(idx)

        # Crop the logits to only the top k options
        logits = torch.from_dlpack(logits)
        logits = logits[:, token_idx - 1, :] / TEMPERATURE
        v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

        # Convert logits to normalized probabilities
        probs = torch.softmax(logits, dim=-1)
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
        idx_next = idx_next.reshape(1, 1)

        # Append sampled index to the running sequence and continue
        idx = torch.from_dlpack(idx).to(torch.int32)
        idx = torch.concat([idx, idx_next], dim=1).to(torch.int32)
        idx = tp.Tensor(idx, device=tp.device("gpu"))

    response = encoder.decode(torch.from_dlpack(idx[0, :]).tolist())
    end_time = time.perf_counter()
    print(f"Generating {args.max_new_tokens} tokens took {end_time - start_time} seconds.")
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
