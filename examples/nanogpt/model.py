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
    dtype: "tripy.datatype" = tp.float32
    quant_mode: str = None


def linear_layer(config: GPTConfig, in_feat, out_feat, bias):
    quant_kwargs = {}
    if config.quant_mode == "int8-weight-only":
        quant_kwargs["quant_dtype"] = tp.int8
        quant_kwargs["weight_quant_dim"] = 0
    elif config.quant_mode == "int4-weight-only":
        quant_kwargs["quant_dtype"] = tp.int4
        quant_kwargs["weight_quant_dim"] = None
    elif config.quant_mode == "fp8":
        quant_kwargs["quant_dtype"] = tp.float8
        quant_kwargs["weight_quant_dim"] = None

    return tp.Linear(
        in_feat,
        out_feat,
        bias=bias,
        dtype=config.dtype,
        **quant_kwargs,
    )


class CausalSelfAttention(tp.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_size % config.num_heads == 0
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.num_heads = config.num_heads
        self.embedding_size = config.embedding_size
        self.c_attn = linear_layer(config, config.embedding_size, 3 * config.embedding_size, config.bias)
        self.c_proj = linear_layer(config, config.embedding_size, config.embedding_size, config.bias)
        self.bias = tp.tril(tp.ones((config.block_size, config.block_size), dtype=config.dtype))

    def __call__(self, x: tp.Tensor, attention_mask: Optional[tp.Tensor] = None):
        qkv = self.c_attn(x)  # (batch_size, seq_len, 3 * embedding_size)

        # WAR for better accuracy and avoid TRT compilation error in fp16
        if self.c_attn.quant_dtype in (tp.float8, tp.int4):
            qkv = tp.cast(qkv, tp.float32)

        q, k, v = tp.split(qkv, 3, dim=2)
        multi_heads = lambda x: tp.transpose(
            tp.reshape(x, (self.batch_size, self.seq_len, self.num_heads, self.embedding_size // self.num_heads)), 1, 2
        )
        q = multi_heads(q)
        k = multi_heads(k)
        v = multi_heads(v)

        k_t = tp.transpose(k, -2, -1)
        att = (q @ k_t) * (1.0 / math.sqrt(self.embedding_size // self.num_heads))

        att = tp.masked_fill(
            att,
            self.bias[: self.seq_len, : self.seq_len] == tp.zeros((self.seq_len, self.seq_len), dtype=self.bias.dtype),
            float("-inf"),
        )

        if attention_mask is not None:
            att = tp.masked_fill(att, attention_mask == 0.0, float("-inf"))

        att = tp.softmax(att, dim=-1)

        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_size) -> (batch_size, num_heads, seq_len, head_size)
        out = att @ v
        out = tp.cast(out, x.dtype)
        out = tp.reshape(tp.transpose(out, 1, 2), (self.batch_size, self.seq_len, self.embedding_size))
        out = self.c_proj(out)  # (batch_size, seq_len, embedding_size)
        return out


class MLP(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = linear_layer(config, config.embedding_size, 4 * config.embedding_size, config.bias)
        self.c_proj = linear_layer(config, 4 * config.embedding_size, config.embedding_size, config.bias)

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
        x_ln1 = tp.cast(self.ln_1(tp.cast(x, self.ln_1.dtype)), x.dtype)
        x = x + self.attn(x_ln1, attention_mask)
        x_ln2 = tp.cast(self.ln_2(tp.cast(x, self.ln_2.dtype)), x.dtype)
        x = x + self.mlp(x_ln2)
        return x


class Transformer(tp.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = tp.Embedding(config.vocab_size, config.embedding_size, dtype=config.dtype)
        self.wpe = tp.Embedding(config.block_size, config.embedding_size, dtype=config.dtype)
        self.h = [Block(config) for _ in range(config.num_layers)]
        self.ln_f = tp.LayerNorm(config.embedding_size)
        self.pos = tp.arange(0, config.seq_len, dtype=tp.int32)

    def __call__(self, idx, attention_mask: Optional[tp.Tensor] = None):
        tok_emb = self.wte(idx)  # token embeddings of shape (batch_size, seq_len, embedding_size)
        pos_emb = self.wpe(self.pos)  # position embeddings of shape (seq_len, embedding_size)
        x = tok_emb + pos_emb  # (batch_size, seq_len, embedding_size)
        for block in self.h:
            x = block(x, attention_mask)
        x = tp.cast(self.ln_f(tp.cast(x, self.ln_f.dtype)), x.dtype)
        return x


class GPT(tp.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert (
            config.seq_len <= config.block_size
        ), f"Cannot forward sequence of length {config.seq_len}, block size is only {config.block_size}"

        self.transformer = Transformer(config)
        if config.quant_mode == "fp8":
            self.lm_head = linear_layer(config, config.embedding_size, config.vocab_size, bias=False)
        else:
            # lm_head is disabled for int8 quantization
            self.lm_head = tp.Linear(config.embedding_size, config.vocab_size, bias=False, dtype=config.dtype)

    # Decorating a function with tp.jit indicates to Tripy that it should compile an optimized
    # version of the implementation the first time the function is called. Subsequent calls will
    # use the faster implementation instead.
    @tp.jit
    def __call__(self, idx, attention_mask: Optional[tp.Tensor] = None):
        x = self.transformer(idx, attention_mask)
        logits = self.lm_head(x)  # (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, vocab_size)
        return logits
