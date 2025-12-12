# RUN: %pick-one-gpu %mlir-trt-jax-py %s

from pathlib import Path
import sys
import pytest
import jax
import numpy as np

mtrt_ops_path = Path(__file__).parent.parent.parent.parent / "python"
assert mtrt_ops_path.exists() and mtrt_ops_path.is_dir()
sys.path.append(str(mtrt_ops_path))

from mlir_tensorrt_jax.flashinfer import single_decode_with_kv_cache  # type: ignore


@pytest.mark.requires_flashinfer
@pytest.mark.requires_compute_capability("ge", major=10)
def test_single_decode_with_kv_cache():
    """Test single decode with KV cache.

    Requires compute capability >= 10.0
    """

    @jax.jit
    def run_single_decode():
        key = jax.random.key(0)
        q = jax.random.normal(key, (32, 128), dtype=np.float16)
        k = jax.random.normal(key, (128, 32, 128), dtype=np.float16)
        v = jax.random.normal(key, (128, 32, 128), dtype=np.float16)
        return single_decode_with_kv_cache(q, k, v)

    out, q_out, k_out = run_single_decode()

    print(f"Output shape: {out.shape}")
    print(f"Output:\n{out}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
