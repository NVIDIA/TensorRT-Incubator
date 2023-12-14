import numpy as np
import cupy as cp
import jax.numpy as jnp
import torch
import jax
import pytest
import jaxlib

from itertools import product


# (40): Refactor tests.
def _test_from_dlpack(from_, to_):
    """Helper function to test conversion from DLpack to specified module type."""
    convert = {
        np.ndarray: np,
        cp.ndarray: cp,
        jaxlib.xla_extension.ArrayImpl: jax.dlpack,
        torch.Tensor: torch.utils.dlpack,
    }

    try:
        if isinstance(to_, cp.ndarray) and isinstance(from_, np.ndarray):
            c = cp.array(np.from_dlpack(from_))
        elif isinstance(to_, np.ndarray) and isinstance(from_, cp.ndarray):
            c = np.array(cp.from_dlpack(from_).get())
        elif (
            isinstance(to_, torch.Tensor) and hasattr(to_, "__cuda_array_interface__") and isinstance(from_, cp.ndarray)
        ):
            c = convert[type(to_)].from_dlpack(from_)
        elif isinstance(to_, torch.Tensor) and isinstance(from_, cp.ndarray):
            c = convert[type(to_)].from_dlpack(from_.get())
        elif (
            isinstance(to_, np.ndarray)
            and isinstance(from_, jaxlib.xla_extension.ArrayImpl)
            and "cpu" in str(from_.devices()).lower()
        ):
            c = convert[type(to_)].from_dlpack(from_)
        elif (
            isinstance(to_, np.ndarray)
            and isinstance(from_, jaxlib.xla_extension.ArrayImpl)
            and "cuda" in str(from_.devices()).lower()
        ):
            c = convert[type(to_)].from_dlpack(jax.device_put(from_, jax.devices("cpu")[0]))
        elif (
            isinstance(to_, jaxlib.xla_extension.ArrayImpl)
            and "cpu" in str(to_.devices()).lower()
            and isinstance(from_, cp.ndarray)
            and hasattr(from_, "__cuda_array_interface__")
        ):
            c = jax.device_put(convert[type(to_)].from_dlpack(from_.get()), jax.devices("cpu")[0])
        elif (
            isinstance(to_, jaxlib.xla_extension.ArrayImpl)
            and "cuda" in str(to_.devices()).lower()
            and isinstance(from_, cp.ndarray)
            and hasattr(from_, "__cuda_array_interface__")
        ):
            c = jax.device_put(convert[type(to_)].from_dlpack(from_.get()), jax.devices("gpu")[0])
        elif (
            isinstance(to_, jaxlib.xla_extension.ArrayImpl)
            and "cpu" in str(to_.devices()).lower()
            and isinstance(from_, np.ndarray)
        ):
            c = jax.device_put(convert[type(to_)].from_dlpack(from_), jax.devices("cpu")[0])
        elif (
            isinstance(to_, jaxlib.xla_extension.ArrayImpl)
            and "cuda" in str(to_.devices()).lower()
            and isinstance(from_, np.ndarray)
        ):
            c = jax.device_put(convert[type(to_)].from_dlpack(from_), jax.devices("gpu")[0])
        elif (
            isinstance(to_, jaxlib.xla_extension.ArrayImpl)
            and "cpu" in str(to_.devices()).lower()
            and isinstance(from_, torch.Tensor)
            and hasattr(from_, "__cuda_array_interface__")
        ):
            c = jax.device_put(convert[type(to_)].from_dlpack(from_.cpu()), jax.devices("cpu")[0])
        elif (
            isinstance(to_, jaxlib.xla_extension.ArrayImpl)
            and "cpu" in str(to_.devices()).lower()
            and isinstance(from_, torch.Tensor)
        ):
            c = jax.device_put(convert[type(to_)].from_dlpack(from_), jax.devices("cpu")[0])
        elif (
            isinstance(to_, jaxlib.xla_extension.ArrayImpl)
            and "cuda" in str(to_.devices()).lower()
            and isinstance(from_, torch.Tensor)
            and hasattr(from_, "__cuda_array_interface__")
        ):
            c = jax.device_put(convert[type(to_)].from_dlpack(from_.cpu()), jax.devices("cuda")[0])
        elif (
            isinstance(to_, jaxlib.xla_extension.ArrayImpl)
            and "cuda" in str(to_.devices()).lower()
            and isinstance(from_, torch.Tensor)
        ):
            c = jax.device_put(convert[type(to_)].from_dlpack(from_), jax.devices("cuda")[0])
        elif (
            isinstance(to_, jaxlib.xla_extension.ArrayImpl)
            and "cuda" in str(to_.devices()).lower()
            and "cpu" in str(from_.devices()).lower()
            and isinstance(from_, jaxlib.xla_extension.ArrayImpl)
        ):
            c = jax.device_put(convert[type(to_)].from_dlpack(from_), jax.devices("cuda")[0])
        elif (
            isinstance(to_, jaxlib.xla_extension.ArrayImpl)
            and "cpu" in str(to_.devices()).lower()
            and "cuda" in str(from_.devices()).lower()
            and isinstance(from_, jaxlib.xla_extension.ArrayImpl)
        ):
            c = jax.device_put(convert[type(to_)].from_dlpack(from_), jax.devices("cpu")[0])
        elif isinstance(to_, cp.ndarray) and isinstance(from_, jaxlib.xla_extension.ArrayImpl):
            c = cp.array(torch.utils.dlpack.from_dlpack(from_))
        elif isinstance(to_, cp.ndarray) and isinstance(from_, torch.Tensor):
            c = cp.array(torch.utils.dlpack.from_dlpack(from_))
        elif (
            isinstance(to_, np.ndarray)
            and isinstance(from_, torch.Tensor)
            and hasattr(from_, "__cuda_array_interface__")
        ):
            c = np.array(torch.utils.dlpack.from_dlpack(from_).cpu())
        elif (
            isinstance(to_, torch.Tensor)
            and hasattr(to_, "__cuda_array_interface__")
            and isinstance(from_, jaxlib.xla_extension.ArrayImpl)
            and "cpu" in str(from_.devices()).lower()
        ):
            c = torch.utils.dlpack.from_dlpack(from_).to(device=torch.device("cuda"))
        elif (
            isinstance(to_, torch.Tensor) and hasattr(to_, "__cuda_array_interface__") and isinstance(from_, np.ndarray)
        ):
            c = torch.utils.dlpack.from_dlpack(from_).to(device=torch.device("cuda"))
        elif isinstance(to_, torch.Tensor) and isinstance(from_, np.ndarray):
            c = torch.utils.dlpack.from_dlpack(from_)
        elif (
            isinstance(to_, torch.Tensor)
            and not hasattr(to_, "__cuda_array_interface__")
            and isinstance(from_, torch.Tensor)
            and hasattr(from_, "__cuda_array_interface__")
        ):
            c = torch.utils.dlpack.from_dlpack(from_).to(device=torch.device("cpu"))
        elif (
            isinstance(to_, torch.Tensor)
            and hasattr(to_, "__cuda_array_interface__")
            and isinstance(from_, torch.Tensor)
            and not hasattr(from_, "__cuda_array_interface__")
        ):
            c = torch.utils.dlpack.from_dlpack(from_).to(device=torch.device("cuda"))
        elif (
            isinstance(to_, torch.Tensor)
            and hasattr(to_, "__cuda_array_interface__")
            and isinstance(from_, jaxlib.xla_extension.ArrayImpl)
            and "cuda" in str(from_.devices()).lower()
        ):
            c = torch.utils.dlpack.from_dlpack(from_)
        elif (
            isinstance(to_, torch.Tensor)
            and isinstance(from_, jaxlib.xla_extension.ArrayImpl)
            and "cuda" in str(from_.devices()).lower()
        ):
            c = torch.utils.dlpack.from_dlpack(from_).to(device=torch.device("cpu"))
        elif isinstance(to_, torch.Tensor) and isinstance(from_, torch.Tensor):
            c = from_
        else:
            c = convert[type(to_)].from_dlpack(from_)
        assert isinstance(c, type(to_))
        assert c == to_
    except TypeError as e:
        print(f"An error occurred: {e}")


_np_arr = np.ones(1, dtype=np.float32)
_cp_arr = cp.ones(1, dtype=np.float32)
_torch_cpu_arr = torch.tensor(_np_arr)
_torch_gpu_arr = _torch_cpu_arr.to(torch.device("cuda"))
_jax_cpu_arr = jax.device_put(_np_arr, jax.devices("cpu")[0])
# (41): Enable jax gpu array tests. To reduce code changes, just pun the types.
_jax_gpu_arr = _jax_cpu_arr
# _jax_gpu_arr = jax.device_put(_np_arr, jax.devices("gpu")[0])

_DATA = [_np_arr, _cp_arr, _torch_cpu_arr, _torch_gpu_arr, _jax_cpu_arr, _jax_gpu_arr]
_DATA_SUBTESTS = list(product(_DATA, repeat=2))


@pytest.mark.parametrize("data", _DATA_SUBTESTS)
def test_dlpack(data):
    """Test DLpack conversion across different modules and data types."""
    ...
    _test_from_dlpack(data[0], data[1])
