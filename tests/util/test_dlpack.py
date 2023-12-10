import numpy as np
import cupy as cp
import jax.numpy as jnp
import torch
import jax
import pytest

# Helper functions for DLpack conversion testing


def _from_dlpack(from_, to_, _module):
    """Helper function to test conversion from DLpack to specified module type."""
    v = _module.from_dlpack(from_)
    assert isinstance(v, type(to_))
    assert v == to_


def _cp_from_dlpack(from_, to_, _module, _smodule):
    """Helper function to test conversion from DLpack to Cupy array."""
    v = _module.array(_smodule.from_dlpack(from_))
    assert isinstance(v, type(to_))
    assert v == to_


def _np_from_dlpack(from_, to_, _module, _smodule):
    """Helper function to test conversion from DLpack to NumPy array."""
    v = _module.array(_smodule.from_dlpack(from_).get())
    assert isinstance(v, type(to_))
    assert v == to_


def _torch_from_torch_gpu_dlpack(from_, to_, _module):
    """Helper function to test conversion from DLpack to Torch CPU tensor."""
    v = torch.utils.dlpack.from_dlpack(from_).cpu()
    assert isinstance(v, type(to_))
    assert v == to_


def _np_from_torch_gpu_dlpack(from_, to_, _module):
    """Helper function to test conversion from DLpack to Numpy array."""
    v = _module.array(torch.utils.dlpack.from_dlpack(from_).cpu())
    assert isinstance(v, type(to_))
    assert v == to_


def _torch_gpu_from_torch_dlpack(from_, to_, _module):
    """Helper function to test conversion from DLpack to Torch GPU tensor."""
    v = _module.from_dlpack(from_).to(torch.device("cuda"))
    assert isinstance(v, type(to_))
    assert v == to_


def _torch_from_array_dlpack(from_, to_, _module, device):
    """Helper function to test conversion from DLpack to Torch tensor on specified device."""
    v = _module.from_dlpack(from_).to(device)
    assert isinstance(v, type(to_))
    assert v == to_


def _jax_gpu_from_dlpack(from_, to_, _module):
    """Helper function to test conversion from DLpack to Jax GPU tensor."""
    v = jax.device_put(_module.from_dlpack(from_), jax.devices("gpu")[0])
    assert isinstance(v, type(to_))
    assert v == to_


# Main test function for DLpack conversion


@pytest.mark.parametrize("dtype", [np.float32])
def test_dlpack(dtype):
    """Test DLpack conversion across different modules and data types."""
    np_arr = np.ones(1, dtype=dtype)
    cp_arr = cp.ones(1, dtype=dtype)
    torch_arr = torch.tensor(np_arr)
    torch_gpu_arr = torch_arr.to(torch.device("cuda"))
    jax_cpu_arr = jnp.array(np_arr)

    # Convert to numpy array
    _from_dlpack(np_arr, np_arr, np)
    _np_from_dlpack(cp_arr, np_arr, np, cp)
    _from_dlpack(jax_cpu_arr, np_arr, np)
    _from_dlpack(torch_arr, np_arr, np)
    _np_from_torch_gpu_dlpack(torch_gpu_arr, np_arr, np)

    # Convert to cupy array
    _cp_from_dlpack(np_arr, cp_arr, cp, np)
    _from_dlpack(cp_arr, cp_arr, cp)
    _cp_from_dlpack(jax_cpu_arr, cp_arr, cp, jax.dlpack)
    _cp_from_dlpack(torch_arr, cp_arr, cp, torch.utils.dlpack)
    _from_dlpack(torch_gpu_arr, cp_arr, cp)

    # Convert to jax tensor
    _from_dlpack(np_arr, jax_cpu_arr, jax.dlpack)
    # TODO: Enable after GPU backend is enabled for Jax.
    # _jax_gpu_from_dlpack(cp_arr, jax_cpu_arr, jax.dlpack)
    _from_dlpack(jax_cpu_arr, jax_cpu_arr, jax.dlpack)
    _from_dlpack(torch_arr, jax_cpu_arr, jax.dlpack)

    # Convert to torch CPU tensor
    _from_dlpack(np_arr, torch_arr, torch.utils.dlpack)
    _torch_from_array_dlpack(cp_arr, torch_arr, torch.utils.dlpack, "cpu")
    _from_dlpack(jax_cpu_arr, torch_arr, torch.utils.dlpack)
    _from_dlpack(torch_arr, torch_arr, torch.utils.dlpack)
    _torch_from_torch_gpu_dlpack(torch_gpu_arr, torch_arr, torch.utils.dlpack)

    # Convert to torch GPU tensor
    _torch_from_array_dlpack(np_arr, torch_gpu_arr, torch.utils.dlpack, torch.device("cuda"))
    _from_dlpack(cp_arr, torch_gpu_arr, torch.utils.dlpack)
    _torch_gpu_from_torch_dlpack(jax_cpu_arr, torch_gpu_arr, torch.utils.dlpack)
    _torch_gpu_from_torch_dlpack(torch_arr, torch_gpu_arr, torch.utils.dlpack)
    _from_dlpack(torch_gpu_arr, torch_gpu_arr, torch.utils.dlpack)
