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
    seq_len: int = 1
    batch_size: int = 1


class CausalSelfAttention(tp.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_size % config.num_heads == 0
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.num_heads = config.num_heads
        self.embedding_size = config.embedding_size
        self.c_attn = tp.Linear(config.embedding_size, 3 * config.embedding_size, bias=config.bias)
        self.c_proj = tp.Linear(config.embedding_size, config.embedding_size, bias=config.bias)
        self.bias = tp.tril(tp.ones((config.block_size, config.block_size)))

    def __call__(self, x: tp.Tensor, attention_mask: Optional[tp.Tensor] = None):
        attn = self.c_attn(x)  # (batch_size, seq_len, 3 * embedding_size)

        # q, k, v = attn.split(self.embedding_size, dim=2)
        def extract(index):
            weight = attn[:, :, index * self.embedding_size : (index + 1) * self.embedding_size]
            return tp.transpose(
                tp.reshape(
                    weight, (self.batch_size, self.seq_len, self.num_heads, self.embedding_size // self.num_heads)
                ),
                1,
                2,
            )  # (batch_size, num_heads, seq_len, head_size)

        q, k, v = extract(0), extract(1), extract(2)
        k_t = tp.transpose(k, -2, -1)
        att = (q @ k_t) * (1.0 / math.sqrt(self.embedding_size // self.num_heads))

        att = tp.masked_fill(
            att,
            self.bias[: self.seq_len, : self.seq_len] == tp.zeros((self.seq_len, self.seq_len), dtype=tp.float32),
            float("-inf"),
        )

        if attention_mask is not None:
            att = tp.masked_fill(att, attention_mask == 0.0, float("-inf"))

        att = tp.softmax(att, dim=-1)
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_size) -> (batch_size, num_heads, seq_len, head_size)
        out = att @ v
        out = tp.reshape(tp.transpose(out, 1, 2), (self.batch_size, self.seq_len, self.embedding_size))
        out = self.c_proj(out)  # (batch_size, seq_len, embedding_size)
        return out


class MLP(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = tp.Linear(config.embedding_size, 4 * config.embedding_size, bias=config.bias)
        self.c_proj = tp.Linear(4 * config.embedding_size, config.embedding_size, bias=config.bias)

    def __call__(self, x):
        x = self.c_fc(x)
        x = tp.gelu(x)
        x = self.c_proj(x)
        return x


class Block(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = tp.LayerNorm(config.embedding_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = tp.LayerNorm(config.embedding_size)
        self.mlp = MLP(config)

    def __call__(self, x, attention_mask: Optional[tp.Tensor] = None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = tp.Embedding(config.vocab_size, config.embedding_size)
        self.wpe = tp.Embedding(config.block_size, config.embedding_size)
        self.h = [Block(config) for _ in range(config.num_layers)]
        self.ln_f = tp.LayerNorm(config.embedding_size)
        self.pos = tp.arange(0, config.seq_len, dtype=tp.int32)

    def __call__(self, idx, attention_mask: Optional[tp.Tensor] = None):
        tok_emb = self.wte(idx)  # token embeddings of shape (batch_size, seq_len, embedding_size)
        pos_emb = self.wpe(self.pos)  # position embeddings of shape (seq_len, embedding_size)
        x = tok_emb + pos_emb  # (batch_size, seq_len, embedding_size)
        for block in self.h:
            x = block(x, attention_mask)
        return self.ln_f(x)


class GPT(tp.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert (
            config.seq_len <= config.block_size
        ), f"Cannot forward sequence of length {config.seq_len}, block size is only {config.block_size}"

        self.transformer = Transformer(config)
        self.lm_head = tp.Linear(config.embedding_size, config.vocab_size, bias=False)

    # Decorating a function with tp.jit indicates to Tripy that it should compile an optimized
    # version of the implementation the first time the function is called. Subsequent calls will
    # use the faster implementation instead.
    @tp.jit
    def __call__(self, idx, attention_mask: Optional[tp.Tensor] = None):
        x = self.transformer(idx, attention_mask)

        logits = self.lm_head(x)  # (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, vocab_size)
        return logits
