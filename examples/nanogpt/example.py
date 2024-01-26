import argparse
import tiktoken
import torch
from torch.nn import functional as F

from model import GPTConfig, GPT
from weight_loader import load_weights_from_hf
import tripy as tp


def initialize_gpt_model(model_type, token_idx):

    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    }[model_type]

    config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
    config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
    config_args["bias"] = True  # always True for GPT model checkpoints
    config_args["T"] = token_idx
    config_args["B"] = 1

    config = GPTConfig(**config_args)
    model = GPT(config)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nanoGPT sample")

    # Adding the arguments
    parser.add_argument(
        "--start-sequence", type=str, help="The start sequence", default="Do people really like using ONNX?"
    )
    parser.add_argument("--model-type", type=str, help="The model type", default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, help="The maximum number of new tokens", default=10)

    # Parse the arguments
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    start_ids = encode(args.start_sequence)
    B = 1
    T = len(start_ids)
    temperature = 0.8
    top_k = 200

    idx = tp.Tensor(start_ids, dtype=tp.int32).reshape((1, len(start_ids)))

    for token_idx in range(len(start_ids), len(start_ids) + args.max_new_tokens):
        model = initialize_gpt_model(args.model_type, token_idx)
        load_weights_from_hf(model, args.model_type)

        # if the sequence context is growing too long we must crop it at block_size
        # forward the model to get the logits for the index in the sequence
        logits = model(idx)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        logits = torch.Tensor(logits.numpy())
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((torch.Tensor(idx.numpy()), idx_next), dim=1).to(torch.int32)
        idx = tp.Tensor(idx, device=tp.device("gpu"))
        T = T + 1

    print(decode(idx[0].numpy().tolist()))
