"""Test NVFP4 quantization-dequantization nodes' JAX implementation in MLIR-TensorRT.

The test calculates the quantization error, defined as the difference between
tensor x and tensor dq(q(x)). The error tolerance is set to abs error: 0.2 or
rel error: 0.25.
"""

# REQUIRES: all-gpus-support-fp4
# REQUIRES: has-support-for-ptx-gte-86
# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def blockQuantizeNVFP4(x: jnp.ndarray, tensor_amax: jnp.float32):
    assert x.shape == (16,)
    assert x.dtype == jnp.float32
    block_amax = jnp.max(jnp.abs(x))
    S_dec_x = block_amax / 6.0
    S_enc = jnp.float32(6.0 * 448.0 / tensor_amax)
    S_dec_x_e4m3 = jnp.float8_e4m3fn(S_dec_x * S_enc)
    S_enc_x = S_enc / jnp.float32(S_dec_x_e4m3)
    qx = (x * S_enc_x).astype(jnp.float4_e2m1fn)
    return qx, S_dec_x_e4m3


def blockDequantizeNVFP4(
    qx: jnp.ndarray, S_dec_x_e4m3: jnp.float8_e4m3fn, tensor_amax: jnp.float32
):
    assert qx.shape == (16,)
    assert qx.dtype == jnp.float4_e2m1fn
    S_dec = jnp.float32(tensor_amax / (6.0 * 448.0))
    S_dec_unscaled = jnp.float32(S_dec_x_e4m3) * S_dec
    return qx.astype(jnp.float32) * S_dec_unscaled


@jax.jit
def quantizeNVFP4(tensor: jnp.ndarray):
    assert tensor.dtype == jnp.float32
    original_shape = tensor.shape
    assert (
        original_shape[-1] % 16 == 0
    ), f"NVFP4 needs the tensor's last dimension to be a multiple of 16 but got {original_shape}."
    tensor_amax = jnp.max(jnp.abs(tensor))

    quantized_tensor, S_dec_blocks_e4m3 = jax.vmap(
        blockQuantizeNVFP4, in_axes=(0, None)
    )(tensor.reshape((-1, 16)), tensor_amax)
    quantized_tensor = quantized_tensor.reshape(original_shape)
    S_dec_blocks_e4m3 = S_dec_blocks_e4m3.reshape(
        original_shape[:-1] + (original_shape[-1] // 16,)
    )
    return quantized_tensor, S_dec_blocks_e4m3, tensor_amax


@jax.jit
def dequantizeNVFP4(
    quantized_tensor: jnp.ndarray,
    S_dec_blocks_e4m3: jnp.ndarray,
    tensor_amax: jnp.float32,
):
    assert quantized_tensor.dtype == jnp.float4_e2m1fn
    assert S_dec_blocks_e4m3.dtype == jnp.float8_e4m3fn
    original_shape = quantized_tensor.shape
    assert (
        original_shape[-1] % 16 == 0
    ), f"NVFP4 needs the tensor's last dimension to be a multiple of 16 but got {original_shape}."
    tensor = jax.vmap(blockDequantizeNVFP4, in_axes=(0, 0, None))(
        quantized_tensor.reshape((-1, 16)),
        S_dec_blocks_e4m3.reshape(-1, 1),
        tensor_amax,
    )
    return tensor.reshape(original_shape)


@pytest.mark.requires_fp4
@pytest.mark.requires_minimum_ptx_version(version=86)
def test_q_dq_nvfp4():
    """Test NVFP4 quantization and dequantization with various tensor shapes."""
    key = jax.random.PRNGKey(26)
    for shape in [(1, 16), (3, 64), (2, 128, 16), (1, 7, 100, 32), (2, 256, 32, 128)]:
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape, dtype=jnp.float32)
        result = dequantizeNVFP4(*quantizeNVFP4(x))
        np.testing.assert_allclose(result, x, atol=0.2, rtol=0.25)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
