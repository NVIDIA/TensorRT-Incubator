import argparse

import tiktoken
import torch
from model import GPT, GPTConfig
from weight_loader import load_weights_from_hf

import tripy as tp


def initialize_gpt_model(model_type, padded_seq_len, dtype):
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

    args = parser.parse_args()

    encoder = tiktoken.get_encoding("gpt2")

    input_ids = encoder.encode(args.input_text, allowed_special={"<|endoftext|>"})

    TEMPERATURE = 0.8
    TOP_K = 200

    padded_seq_len = len(input_ids) + args.max_new_tokens

    model_dtype = tp.float16

    model = initialize_gpt_model(args.model_type, padded_seq_len, model_dtype)
    load_weights_from_hf(model, args.model_type, model_dtype)

    idx = torch.Tensor(input_ids).reshape((1, len(input_ids)))
    zeros = torch.zeros((1, args.max_new_tokens), dtype=torch.float32)
    idx = tp.copy(tp.Tensor(torch.cat((idx, zeros), dim=1).to(torch.int32)), tp.device("gpu"))

    def generate_attention_mask(input_tokens):
        # Check where input_tokens !=0 and fill with ones.
        zeros = tp.zeros_like(input_tokens)
        ones = tp.ones_like(input_tokens)
        return tp.reshape(tp.cast(tp.where(input_tokens == 0, zeros, ones), model_dtype), (1, 1, 1, padded_seq_len))

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed)

    for token_idx in range(len(input_ids), len(input_ids) + args.max_new_tokens):
        mask = generate_attention_mask(idx)
        logits = model(idx, mask)

        # Scale by desired temperature
        logits = logits[:, token_idx - 1, :] / TEMPERATURE

        # Crop the logits to only the top k options
        logits = torch.from_dlpack(logits)
        v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

        # Convert logits to normalized probabilities
        probs = torch.softmax(logits, dim=-1)
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1, generator=generator).cpu()

        # Append sampled index to the running sequence and continue
        idx = torch.from_dlpack(idx).to(torch.int32)
        idx[0, token_idx] = idx_next[0]
        idx = tp.Tensor(idx, device=tp.device("gpu"))

    response = encoder.decode(idx[0].numpy().tolist())
    print(response)


if __name__ == "__main__":
    main()
