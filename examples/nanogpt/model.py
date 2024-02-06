import math
from dataclasses import dataclass
from typing import Optional

import tripy as tp


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    embedding_size: int = 768
    bias: bool = True
    T: int = 1
    B: int = 1


class CausalSelfAttention(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_size % config.num_heads == 0
        self.T = config.T
        self.B = config.B
        self.num_heads = config.num_heads
        self.embedding_size = config.embedding_size
        self.c_attn = tp.nn.Linear(config.embedding_size, 3 * config.embedding_size, bias=config.bias)
        self.c_proj = tp.nn.Linear(config.embedding_size, config.embedding_size, bias=config.bias)
        self.bias = tp.ones((config.block_size, config.block_size)).tril()

    def __call__(self, x: tp.Tensor, attention_mask: Optional[tp.Tensor] = None):
        E = self.embedding_size
        attn = self.c_attn(x)  # (B, T, 3 * E)

        # q, k, v = attn.split(self.embedding_size, dim=2)
        def extract(index):
            weight = attn[:, :, index * E : (index + 1) * E]
            return weight.reshape((self.B, self.T, self.num_heads, E // self.num_heads)).transpose(
                1, 2
            )  # (B, nh, T, hs)

        q, k, v = extract(0), extract(1), extract(2)
        k_t = k.transpose(-2, -1)
        att = (q @ k_t) * (1.0 / math.sqrt(E // self.num_heads))

        att = att.masked_fill(
            self.bias[: self.T, : self.T] == tp.zeros((self.T, self.T), dtype=tp.float32),
            float("-inf"),
        )

        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0.0, float("-inf"))

        att = tp.nn.softmax(att, dim=-1)
        out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).reshape((self.B, self.T, E))
        out = self.c_proj(out)  # (B, T, E)
        return out


class MLP(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = tp.nn.Linear(config.embedding_size, 4 * config.embedding_size, bias=config.bias)
        self.c_proj = tp.nn.Linear(4 * config.embedding_size, config.embedding_size, bias=config.bias)

    def __call__(self, x):
        x = self.c_fc(x)
        x = tp.nn.gelu(x)
        x = self.c_proj(x)
        return x


class Block(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = tp.nn.LayerNorm(config.embedding_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = tp.nn.LayerNorm(config.embedding_size)
        self.mlp = MLP(config)

    def __call__(self, x, attention_mask: Optional[tp.Tensor] = None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = tp.nn.Embedding(config.vocab_size, config.embedding_size)
        self.wpe = tp.nn.Embedding(config.block_size, config.embedding_size)
        # (#99): Below 2 lines will become self.h = [Block(config) for _ in range(config.num_layers)]
        self.num_layers = config.num_layers
        for i in range(self.num_layers):
            setattr(self, f"h_{i}", Block(config))
        self.ln_f = tp.nn.LayerNorm(config.embedding_size)
        self.pos = tp.arange(0, config.T, dtype=tp.int32)  # shape (t)

    def __call__(self, idx, attention_mask: Optional[tp.Tensor] = None):
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, embedding_size)
        pos_emb = self.wpe(self.pos)  # position embeddings of shape (t, embedding_size)
        x = tok_emb + pos_emb  # (B, T, E)
        for i in range(self.num_layers):
            x = getattr(self, f"h_{i}")(x, attention_mask)
        return self.ln_f(x)


class GPT(tp.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert (
            config.T <= config.block_size
        ), f"Cannot forward sequence of length {config.T}, block size is only {config.block_size}"

        self.transformer = Transformer(config)
        self.lm_head = tp.nn.Linear(config.embedding_size, config.vocab_size, bias=False)

    # Decorating a function with tp.jit indicates to Tripy that it should compile an optimized
    # version of the implementation the first time the function is called. Subsequent calls will
    # use the faster implementation instead.
    @tp.jit
    def __call__(self, idx, attention_mask: Optional[tp.Tensor] = None):
        # forward the GPT model itself
        x = self.transformer(idx, attention_mask)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x)  # (B, T, E) -> (B, T, vocab_size)
        return logits
