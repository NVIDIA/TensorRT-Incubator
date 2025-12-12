# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import itertools
from pathlib import Path
import sys
import pytest
import jax
import jax.numpy as jnp

mtrt_ops_path = Path(__file__).parent.parent.parent.parent / "python"
assert mtrt_ops_path.exists() and mtrt_ops_path.is_dir()
sys.path.append(str(mtrt_ops_path))

from mlir_tensorrt_jax.flashinfer import single_prefill_with_kv_cache  # type: ignore
from mlir_tensorrt_jax.flashinfer.utils import is_float8  # type: ignore


# Test parameters
q_dtypes = [jnp.float16, jnp.bfloat16]
kv_dtypes = [jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn]
causals = [True, False]
window_lefts = [-1, 64]

# Generate all test parameter combinations
test_params = list(itertools.product(q_dtypes, kv_dtypes, causals, window_lefts))


@pytest.mark.requires_flashinfer
@pytest.mark.requires_compute_capability("ge", major=10)
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,causal,window_left",
    test_params,
    ids=[
        f"q_{q_dtype.dtype.name}_kv_{kv_dtype.dtype.name}_causal_{causal}_win_{window_left}"
        for q_dtype, kv_dtype, causal, window_left in test_params
    ],
)
def test_single_prefill_with_kv_cache(q_dtype, kv_dtype, causal, window_left):
    """Test single prefill with KV cache for various dtype and configuration combinations.

    Requires compute capability >= 10.0
    """
    # Skip unsupported configurations
    if not is_float8(kv_dtype):
        if kv_dtype != q_dtype:
            pytest.skip(
                f"Unsupported: q_dtype={q_dtype} != kv_dtype={kv_dtype} (non-float8)"
            )

    def run_single_prefill():
        key = jax.random.key(0)
        q = jax.random.normal(key, (128, 32, 128), dtype=q_dtype)
        k = jax.random.normal(key, (128, 32, 128), dtype=kv_dtype)
        v = jax.random.normal(key, (128, 32, 128), dtype=kv_dtype)
        return single_prefill_with_kv_cache(
            q, k, v, causal=causal, window_left=window_left
        )

    out, q_out, k_out = jax.jit(run_single_prefill)()
    out.block_until_ready()

    print(
        f"SUCCESS: q_dtype={q_dtype.dtype.name}, kv_dtype={kv_dtype.dtype.name}, "
        f"causal={causal}, window_left={window_left}, output_shape={out.shape}"
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
