import math
from dataclasses import dataclass

import tripy as tp


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True
    T: int = 1
    B: int = 1


class CausalSelfAttention(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
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
            return weight.reshape((self.config.B, self.config.T, self.n_head, E // self.n_head)).transpose(
                1, 2
            )  # (B, nh, T, hs)

        q, k, v = extract(0), extract(1), extract(2)
        k_t = k.transpose(-2, -1)
        att = (q @ k_t) * (1.0 / math.sqrt(E // self.n_head))
        att = att.masked_fill(
            self.bias[: self.config.T, : self.config.T] == tp.zeros((self.config.T, self.config.T), dtype=tp.float32),
            float("-inf"),
        )
        att = tp.nn.softmax(att, dim=-1)
        out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).reshape((self.config.B, self.config.T, E))
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
            self.config.T <= self.config.block_size
        ), f"Cannot forward sequence of length {self.config.T}, block size is only {self.config.block_size}"
        pos = tp.arange(0, self.config.T, dtype=tp.int32)  # shape (t)

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
