import argparse
import torch
import re
import math
from dataclasses import dataclass
import tiktoken
import tempfile
from torch.nn import functional as F

import tripy as tp

temperature = 0.8
top_k = 200


class CausalSelfAttention(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = tp.nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = tp.nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.bias = tp.tril(tp.ones((config.block_size, config.block_size)))

    def __call__(self, x: tp.Tensor):
        E = self.n_embd
        attn = self.c_attn(x)  # (B, T, 3 * E)

        # q, k, v = attn.split(self.n_embd, dim=2)
        def extract(index):
            weight = attn[:, :, index * E : (index + 1) * E]
            return weight.reshape((B, T, self.n_head, E // self.n_head)).transpose(1, 2)  # (B, nh, T, hs)

        q, k, v = extract(0), extract(1), extract(2)
        k_t = k.transpose(-2, -1)
        att = (q @ k_t) * (1.0 / math.sqrt(E // self.n_head))
        att = att.masked_fill(self.bias[:T, :T] == tp.zeros((T, T), dtype=tp.float32), float("-inf"))
        att = tp.nn.softmax(att, dim=-1)
        out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).reshape((B, T, E))
        out = self.c_proj(out)  # (B, T, E)
        return out


class MLP(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = tp.nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = tp.nn.gelu
        self.c_proj = tp.nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = tp.nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = tp.nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True


class Transformer(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = tp.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = tp.nn.Embedding(config.block_size, config.n_embd)
        # (#99): Below 2 lines will become self.h = [Block(config) for _ in range(config.n_layer)]
        for i in range(config.n_layer):
            setattr(self, f"h_{i}", Block(config))
        self.ln_f = tp.nn.LayerNorm(config.n_embd)


class GPT(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = Transformer(config)
        self.lm_head = tp.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def __call__(self, idx):
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = tp.arange(0, T, dtype=tp.int32)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb  # (B, T, E)
        for i in range(self.config.n_layer):
            x = getattr(self.transformer, f"h_{i}")(x)

        x = self.transformer.ln_f(x)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, -1:, :])  # (B, 1, E) -> (B, 1, vocab_size)
        return logits

    @staticmethod
    def from_pretrained(model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]  # same, just the mask (buffer)
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        sd_hf["transformer.wte.weight"] = sd_hf["lm_head.weight"]  # https://paperswithcode.com/method/weight-tying

        # transformer.h.i weights are stored as transformer.h_i
        converted_list = [re.sub(r"h\.\d+", lambda x: x.group().replace(".", "_"), s) for s in sd_keys_hf]

        for idx, k in enumerate(sd_keys_hf):
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                with torch.no_grad():
                    sd[converted_list[idx]] = tp.nn.Parameter(tp.Tensor(sd_hf[k].t().contiguous()))
            else:
                # vanilla copy over the other parameters
                with torch.no_grad():
                    sd[converted_list[idx]] = tp.nn.Parameter(tp.Tensor(sd_hf[k]))

        model.load_from_state_dict(sd)
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
    idx = tp.Tensor(start_ids, dtype=tp.int32).reshape((1, len(start_ids)))

    for _ in range(args.max_new_tokens):
        model = GPT.from_pretrained(args.model_type)

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
