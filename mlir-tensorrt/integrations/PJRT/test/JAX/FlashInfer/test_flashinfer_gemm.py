# RUN: %pick-one-gpu %mlir-trt-jax-py %s

from pathlib import Path
import sys
import pytest
import jax
import jax.numpy as jnp

mtrt_ops_path = Path(__file__).parent.parent.parent.parent / "python"
assert mtrt_ops_path.exists() and mtrt_ops_path.is_dir()
sys.path.append(str(mtrt_ops_path))

from mlir_tensorrt_jax.flashinfer import bmm_fp8  # type: ignore


@pytest.mark.requires_flashinfer
@pytest.mark.requires_compute_capability("ge", major=10)
@pytest.mark.unsupported_compute_capability("eq", major=12)
def test_bmm_fp8():
    """Test FP8 batch matrix multiplication.

    Requires compute capability >= 10.0
    Known issues on SM120/121 (compute capability 12.x)
    """

    @jax.jit
    def run_bmm_fp8():
        key = jax.random.key(0)
        A = jax.random.normal(key, (1, 1024, 256), dtype=jnp.float8_e4m3fn)
        B = jax.random.normal(key, (1, 256, 1024), dtype=jnp.float8_e4m3fn)
        A_scale = jnp.ones((1, 1, 1), dtype=jnp.float32)
        B_scale = jnp.ones((1, 1, 1), dtype=jnp.float32)
        return bmm_fp8(A, B, A_scale, B_scale, dtype=jnp.float16, tactic=0)

    out = run_bmm_fp8()
    print(f"Output shape: {out.shape}")
    print(f"Output:\n{out}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
