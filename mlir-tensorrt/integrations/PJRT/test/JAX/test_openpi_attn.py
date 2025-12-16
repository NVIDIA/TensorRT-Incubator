"""Test based on the OpenPI attention module from the OpenPI repo.

This is used to test the attention module's key features (Einsum and multi-head
attention) in MLIR-TensorRT. The test is testing Jax backend against the baseline
NumPy implementation. The testcases are gemmas and gemma_loras.
"""

# RUN: %pick-one-gpu %mlir-trt-jax-py %s
# REQUIRES: long_tests

import math
import re
from typing import Any
from collections.abc import Sequence
import dataclasses
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import softmax


def _apply_rope(numpy, x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * numpy.arange(
        x.shape[-1] // 2, dtype=numpy.float32
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    radians = radians.astype(numpy.float32)
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = numpy.sin(radians), numpy.cos(radians)
    x1, x2 = numpy.split(x, 2, axis=-1)
    res = numpy.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    res = res.astype(numpy.float32)
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _make_lora_eqns(lora_config, eqn: str) -> tuple[str, str]:
    if "L" in eqn:
        raise ValueError(f"L already in eqn: {eqn}")
    if not (m := re.match("(.*),(.*)->(.*)", eqn)):
        raise ValueError(f"Unsupported einsum eqn: {eqn}")
    lhs, rhs, out = m.groups()

    assert lora_config is not None
    a_label, b_label = (rhs[x] for x in lora_config.axes)
    label = lora_config.label

    a_rhs = rhs.replace(b_label, label)
    a_out = out.replace(b_label, label)
    eqn_a = f"{lhs},{a_rhs}->{a_out}"

    b_rhs = rhs.replace(a_label, label)
    eqn_b = f"{a_out},{b_rhs}->{out}"

    return eqn_a, eqn_b


@dataclasses.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return (
            self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank
        )


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, LoRAConfig] = dataclasses.field(default_factory=dict)


class EinsumJax(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = _make_lora_eqns(config, eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result


class AttentionJax(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(
            config.head_dim == self.configs[0].head_dim for config in self.configs
        )
        assert all(
            config.num_heads == self.configs[0].num_heads for config in self.configs
        )
        assert all(
            config.num_kv_heads == self.configs[0].num_kv_heads
            for config in self.configs
        )

        dtype = next(
            x.dtype for x in xs if x is not None
        )  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = EinsumJax(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(
                        in_axis=-2, out_axis=-1, batch_axis=(0, 1)
                    ),
                    lora_config=config.lora_configs.get("attn"),
                )
                q, k, v = qkv_einsum("BSD,3KDH->3BSKH", x)
                qkvs.append((q, k, v))
            else:
                q_einsum = EinsumJax(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(
                        in_axis=-2, out_axis=-1, batch_axis=(0,)
                    ),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = EinsumJax(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(
                        in_axis=-2, out_axis=-1, batch_axis=(0, 1)
                    ),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(jnp, q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(jnp, k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(
            q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads
        )
        logits = jnp.einsum(
            "BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32
        )

        if attn_mask.shape != (
            q.shape[0],
            attn_mask.shape[1],
            q.shape[1],
            k.shape[1],
        ):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = EinsumJax(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


class EinsumNumPy(object):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    def __init__(
        self,
        shape: tuple[int, ...],
        lora_config: LoRAConfig | None = None,
    ):
        self.shape = shape
        self.lora_config = lora_config

        self.w = np.ones(self.shape)

        if config := self.lora_config:
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = np.ones(shape_a)
            self.w_b = np.ones(shape_b)

    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        if eqn == "BSD,3KDH->3BSKH":
            q = np.einsum("BSD,KDH->BSKH", x, self.w[0].astype(dtype))
            k = np.einsum("BSD,KDH->BSKH", x, self.w[1].astype(dtype))
            v = np.einsum("BSD,KDH->BSKH", x, self.w[2].astype(dtype))
            result = np.stack([q, k, v], axis=0)
            if config := self.lora_config:
                eqn_a, eqn_b = _make_lora_eqns(config, "BSD,KDH->BSKH")
                lora_q = np.einsum(eqn_a, x, self.w_a[0].astype(dtype))
                lora_q = np.einsum(eqn_b, lora_q, self.w_b[0].astype(dtype))
                lora_k = np.einsum(eqn_a, x, self.w_a[1].astype(dtype))
                lora_k = np.einsum(eqn_b, lora_k, self.w_b[1].astype(dtype))
                lora_v = np.einsum(eqn_a, x, self.w_a[2].astype(dtype))
                lora_v = np.einsum(eqn_b, lora_v, self.w_b[2].astype(dtype))
                lora_result = np.stack([lora_q, lora_k, lora_v], axis=0)
                result = result + lora_result * config.scaling_value
        elif eqn == "BSD,2KDH->2BSKH":
            k = np.einsum("BSD,KDH->BSKH", x, self.w[0].astype(dtype))
            v = np.einsum("BSD,KDH->BSKH", x, self.w[1].astype(dtype))
            result = np.stack([k, v], axis=0)
            if config := self.lora_config:
                eqn_a, eqn_b = _make_lora_eqns(config, "BSD,KDH->BSKH")
                lora_k = np.einsum(eqn_a, x, self.w_a[0].astype(dtype))
                lora_k = np.einsum(eqn_b, lora_k, self.w_b[0].astype(dtype))
                lora_v = np.einsum(eqn_a, x, self.w_a[1].astype(dtype))
                lora_v = np.einsum(eqn_b, lora_v, self.w_b[1].astype(dtype))
                lora_result = np.stack([lora_k, lora_v], axis=0)
                result = result + lora_result * config.scaling_value
        else:
            result = np.einsum(eqn, x, self.w.astype(dtype))
            if config := self.lora_config:
                eqn_a, eqn_b = _make_lora_eqns(config, eqn)
                lora = np.einsum(eqn_a, x, self.w_a.astype(dtype))
                lora = np.einsum(eqn_b, lora, self.w_b.astype(dtype))
                result = result + lora * config.scaling_value

        return result


class AttentionNumPy(object):
    """Attention module."""

    def __init__(self, configs: Sequence[Config], key: Any = None):
        self.configs = configs
        self.einsums = {}
        for i, config in enumerate(self.configs):
            if config.num_kv_heads == config.num_heads:
                self.einsums[_name("qkv_einsum", i)] = EinsumNumPy(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    lora_config=config.lora_configs.get("attn"),
                )
            else:
                self.einsums[_name("q_einsum", i)] = EinsumNumPy(
                    shape=(config.num_heads, config.width, config.head_dim),
                    lora_config=config.lora_configs.get("attn"),
                )
                self.einsums[_name("kv_einsum", i)] = EinsumNumPy(
                    shape=(
                        2,
                        config.num_kv_heads,
                        config.width,
                        config.head_dim,
                    ),
                    lora_config=config.lora_configs.get("attn"),
                )
        for i, config in enumerate(self.configs):
            self.einsums[_name("attn_vec_einsum", i)] = EinsumNumPy(
                shape=(config.num_heads, config.head_dim, config.width),
                lora_config=config.lora_configs.get("attn"),
            )

    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(
            config.head_dim == self.configs[0].head_dim for config in self.configs
        )
        assert all(
            config.num_heads == self.configs[0].num_heads for config in self.configs
        )
        assert all(
            config.num_kv_heads == self.configs[0].num_kv_heads
            for config in self.configs
        )

        dtype = next(
            x.dtype for x in xs if x is not None
        )  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkvs.append(self.einsums[_name("qkv_einsum", i)]("BSD,3KDH->3BSKH", x))
            else:
                q = self.einsums[_name("q_einsum", i)]("BTD,NDH->BTNH", x)
                k, v = self.einsums[_name("kv_einsum", i)]("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (np.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(np, q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(np, k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = np.concatenate([cache_k, k], axis=1)
            v = np.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(
            q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads
        )
        logits = np.einsum("BTKGH,BSKH->BKGTS", q, k)

        if attn_mask.shape != (
            q.shape[0],
            attn_mask.shape[1],
            q.shape[1],
            k.shape[1],
        ):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = np.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = softmax(masked_logits, axis=-1).astype(dtype)

        encoded = np.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out.append(
                    self.einsums[_name("attn_vec_einsum", i)](
                        "BTNH,NHD->BTD", encoded[:, start:end]
                    )
                )
                start = end
            else:
                out.append(None)

        return out, (k, v)


def weight_transfer(params, numpy_attention, target_dtype):
    jax_weights, jax_lora_weights = {}, {}
    for name, value in params["params"].items():
        assert isinstance(value, dict) and "w" in value
        jax_weights[name] = np.array(value["w"]).astype(target_dtype)
        if "lora_a" in value:
            jax_lora_weights[f"{name}_lora_a"] = np.array(value["lora_a"]).astype(
                target_dtype
            )
        if "lora_b" in value:
            jax_lora_weights[f"{name}_lora_b"] = np.array(value["lora_b"]).astype(
                target_dtype
            )
    for weight_name in jax_weights.keys():
        assert weight_name in numpy_attention.einsums
        numpy_attention.einsums[weight_name].w = jax_weights[weight_name]
    for weight_name in jax_lora_weights.keys():
        base_name = weight_name.replace("_lora_a", "").replace("_lora_b", "")
        assert base_name in numpy_attention.einsums
        if weight_name.endswith("_lora_a"):
            numpy_attention.einsums[base_name].w_a = jax_lora_weights[weight_name]
        elif weight_name.endswith("_lora_b"):
            numpy_attention.einsums[base_name].w_b = jax_lora_weights[weight_name]
    return


def get_config(variant):
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": LoRAConfig(rank=32, alpha=32.0),
                "ffn": LoRAConfig(rank=32, alpha=32.0),
            },
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": LoRAConfig(rank=16, alpha=16.0),
                "ffn": LoRAConfig(rank=16, alpha=16.0),
            },
        )
    raise ValueError(f"Unknown variant: {variant}")


@pytest.mark.long_test
def test_openpi_attn():
    """Test OpenPI attention module with various Gemma model configurations."""
    key = jax.random.PRNGKey(72)
    batch_size = 2
    num_past_frames = 2
    for model_name in [
        "dummy",
        "gemma_300m",
        "gemma_300m_lora",
        "gemma_2b",
        "gemma_2b_lora",
    ]:
        model_config = get_config(model_name)
        for dtype, atol in [(jnp.float32, 0.01), (jnp.float16, 0.03)]:
            jax_attention = AttentionJax(configs=[model_config])
            key, subkey = jax.random.split(key)
            xs = [
                jax.random.normal(
                    subkey,
                    (batch_size, model_config.depth, model_config.width),
                    dtype=dtype,
                )
            ]
            positions = jnp.arange(model_config.depth)[None, :]
            attn_mask = jnp.ones(
                (
                    batch_size,
                    model_config.num_kv_heads,
                    model_config.depth,
                    model_config.depth + num_past_frames,
                ),
                dtype=bool,
            )
            key, subkey = jax.random.split(key)
            k_cache = jax.random.normal(
                subkey,
                (
                    batch_size,
                    num_past_frames,
                    model_config.num_kv_heads,
                    model_config.head_dim,
                ),
                dtype=dtype,
            )
            key, subkey = jax.random.split(key)
            v_cache = jax.random.normal(
                subkey,
                (
                    batch_size,
                    num_past_frames,
                    model_config.num_kv_heads,
                    model_config.head_dim,
                ),
                dtype=dtype,
            )

            params = jax_attention.init(
                key, xs, positions, attn_mask, (k_cache, v_cache)
            )

            params = jax.tree.map(
                lambda x: x.astype(dtype) if hasattr(x, "astype") else x, params
            )

            jitted_forward = jax.jit(jax_attention.apply)
            jax_outputs, (jax_k, jax_v) = jitted_forward(
                params, xs, positions, attn_mask, (k_cache, v_cache)
            )

            numpy_attention = AttentionNumPy(configs=[model_config], key=key)

            weight_transfer(params, numpy_attention, np.dtype(dtype))

            xs_np = [np.array(x, dtype=np.dtype(dtype)) for x in xs]
            positions_np = np.array(positions)
            attn_mask_np = np.array(attn_mask)

            numpy_outputs, (numpy_k, numpy_v) = numpy_attention(
                xs_np,
                positions_np,
                attn_mask_np,
                (
                    np.array(k_cache, dtype=np.dtype(dtype)),
                    np.array(v_cache, dtype=np.dtype(dtype)),
                ),
            )

            assert len(jax_outputs) == 1
            assert len(numpy_outputs) == 1

            np.testing.assert_allclose(jax_outputs[0], numpy_outputs[0], atol=atol)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
