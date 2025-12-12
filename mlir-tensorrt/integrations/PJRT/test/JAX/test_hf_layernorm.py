# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import flax
import jax
import numpy as np
from jax import numpy as jnp


def flax_layernorm(npy_input, scale, bias, run_config):
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=0)
    jnp_b = jnp.asarray(bias)
    jnp_s = jnp.asarray(scale)

    flax_layernorm = flax.linen.LayerNorm(
        epsilon=run_config["eps"],
        dtype=run_config["dtype"],
        param_dtype=run_config["dtype"],
        use_bias=True,
    )

    variables = flax_layernorm.init(
        prng_key, jnp.empty(run_config["in_shape"], dtype=run_config["dtype"])
    )

    thawed_variables = flax.core.frozen_dict.unfreeze(variables)
    thawed_variables["params"] = flax.core.frozen_dict.freeze(
        {"bias": jnp_b, "scale": jnp_s}
    )

    def test_flax_layer_norm(variables, inputs):
        outputs = flax_layernorm.apply(variables, inputs)
        return outputs

    test_flax_layer_norm_fn = jax.jit(test_flax_layer_norm)
    flax_outputs = test_flax_layer_norm_fn(thawed_variables, inputs)

    return flax_outputs


def expected_layernorm(x, scale, bias, eps):
    num_feat = x.shape[-1]
    x_sq = np.square(x)  # X^2
    avg_x_sq = np.sum(x_sq, axis=-1) / num_feat  # SUM(X^2)
    avg_x = np.sum(x, axis=-1) / num_feat  # SUM(X)
    var_x = avg_x_sq - np.square(avg_x)

    avg_x_r = avg_x[:, :, np.newaxis]
    avg_x_bcast = np.repeat(avg_x_r, num_feat, axis=-1)

    a_x = x - avg_x_bcast

    b_x = np.sqrt(var_x + eps)
    b_x_reshaped = b_x[:, :, np.newaxis]
    b_x_bcast = np.repeat(b_x_reshaped, num_feat, axis=-1)

    c_x = np.divide(scale, b_x_bcast)  # multiplier

    y = np.multiply(a_x, c_x)
    output = y + bias

    return output


def test_hf_layernorm():
    """Test Flax LayerNorm against numpy implementation."""
    # Experiment Configuration
    run_config = {
        "batch_size": 8,
        "inner_dim_size": 128,
        "hidden_size": 1024,
        "eps": 1e-5,
        "dtype": np.float32,
    }

    run_config["in_shape"] = (
        run_config["batch_size"],
        run_config["inner_dim_size"],
        run_config["hidden_size"],
    )

    npy_input = np.random.random(run_config["in_shape"]).astype(np.float32)
    scale = np.random.randn(run_config["hidden_size"]).astype(np.float32)
    bias = np.random.randn(run_config["hidden_size"]).astype(np.float32)

    expected_outputs = expected_layernorm(npy_input, scale, bias, run_config["eps"])
    actual_outputs = flax_layernorm(npy_input, scale, bias, run_config)

    np.testing.assert_allclose(
        expected_outputs,
        actual_outputs,
        atol=1e-5,
        rtol=1e-6,
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
