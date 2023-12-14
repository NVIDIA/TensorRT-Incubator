import numpy as np
import cupy as cp
import jax.numpy as jnp
import torch
import jax
import pytest
import jaxlib

from itertools import product

_np_arr = np.ones(1, dtype=np.float32)
_cp_arr = cp.ones(1, dtype=np.float32)
_torch_cpu_arr = torch.tensor(_np_arr)
_torch_gpu_arr = _torch_cpu_arr.to(torch.device("cuda"))
_jax_cpu_arr = jax.device_put(_np_arr, jax.devices("cpu")[0])
# (41): Enable jax gpu array tests. To reduce code changes, just pun the types.
_jax_gpu_arr = _jax_cpu_arr
# _jax_gpu_arr = jax.device_put(_np_arr, jax.devices("gpu")[0])

_DATA = [_np_arr, _cp_arr, _torch_cpu_arr, _torch_gpu_arr, _jax_cpu_arr]
_DATA_SUBTESTS = list(product(_DATA, repeat=2))


@pytest.mark.parametrize(("from_", "to_"), _DATA_SUBTESTS)
def test_dlpack_interface(from_, to_):
    """Test framework interoperability using __dlpack__ interface."""
    assert hasattr(from_, "__dlpack__") and hasattr(to_, "__dlpack__")

    def _move_to_gpu(d):
        if hasattr(d, "device") and "cuda" in str(d.device).lower():
            if isinstance(from_, torch.Tensor):
                d = d.to(device="cpu")
            else:
                d = d.get()
        return d

    if isinstance(to_, np.ndarray):
        from_ = _move_to_gpu(from_)
        c = np.array(from_)
    if isinstance(to_, cp.ndarray):
        c = cp.array(from_)
    elif isinstance(to_, torch.Tensor):
        if isinstance(from_, jaxlib.xla_extension.ArrayImpl):
            c = torch.utils.dlpack.from_dlpack(from_).to(to_.device)
        else:
            c = torch.as_tensor(from_).to(device=to_.device.type)
    elif isinstance(to_, jaxlib.xla_extension.ArrayImpl):
        from_ = _move_to_gpu(from_)
        c = jax.device_put(jnp.array(from_), jax.devices("cpu")[0])
    else:
        ValueError("Unsupported tensor type")
    assert isinstance(c, type(to_)), print(f"{type(from_)} ->  {type(to_)} == {type(c)}")
    assert c == to_, print(f"{type(from_)} ->  {type(to_)} == {type(c)}")
