import pytest
import cupy as cp
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import jax

from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor
from tripy.common.device import device
from tripy.flat_ir import FlatIR
from tripy.frontend import Tensor, Dim
from tripy import jit
from tests.helper import all_same


class TestFunctional:
    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_add_two_tensors(self, kind):
        arr = np.array([2, 3], dtype=np.float32)
        a = Tensor(arr, device=device(kind))
        b = Tensor(np.ones(2, dtype=np.float32), device=device(kind))

        c = a + b
        out = c + c
        assert (out.eval().cpu_view(np.float32) == np.array([6.0, 8.0])).all()

    @pytest.mark.parametrize("dim", [Dim(2, min=2, opt=2, max=2)])
    def test_add_two_tensors_dynamic(self, dim):
        arr = np.ones(2, dtype=np.float32)
        a = Tensor(arr, shape=(dim))
        b = Tensor(arr, shape=(dim))

        @jit
        def func(a, b):
            c = a + b
            return c

        out = func(a, b)
        assert (out.eval().cpu_view(np.float32) == np.array([2.0, 2.0])).all()

    def test_multi_output_flat_ir(self):
        arr = np.ones(2, dtype=np.float32)
        a = Tensor(arr)
        b = Tensor(arr)
        c = a + b
        d = c + c
        flat_ir = FlatIR([c, d])

        compiler = FlatIRCompiler()
        with FlatIRExecutor(compiler.compile(flat_ir)) as executor:
            out = executor.execute()
            assert (
                len(out) == 2
                and (out[0].data.cpu_view(np.float32) == np.array([2.0, 2.0])).all()
                and (out[1].data.cpu_view(np.float32) == np.array([4.0, 4.0])).all()
            )

    def _test_framework_interoperability(self, data, device):
        a = Tensor(data, device=device)
        b = Tensor(torch.tensor(data), device=device)

        if device.kind == "gpu":
            # TODO: Enable this when upgrade to 12.2.
            # Also, fix explicit .get() call here. can we construct Jax array from cupy directly.
            c = Tensor(jax.device_put(jnp.array(data.get()), jax.devices("gpu")[0]), device=device)
        else:
            c = Tensor(jax.device_put(jnp.array(data), jax.devices("cpu")[0]))

        out = a + b + c
        assert (out.eval().cpu_view(np.float32) == np.array([3.0, 3.0])).all()

    def test_cpu_and_gpu_framework_interoperability(self):
        from tripy.common.device import device as make_device

        self._test_framework_interoperability(np.ones(2, np.float32), device=make_device("cpu"))
        self._test_framework_interoperability(cp.ones(2, np.float32), device=make_device("gpu"))

    def _assert_round_tripping(self, original_data, tensor, round_trip, compare, data_type=np.float32):
        """Assert round-tripping for different frameworks."""
        round_tripped_data = tensor.op.data.cpu_view(data_type)
        assert (round_tripped_data == original_data).all()
        assert round_tripped_data.data == original_data.data

    def _test_round_tripping(self, data, device):
        assert isinstance(data, np.ndarray) or isinstance(data, cp.ndarray)

        # Assert round-tripping for numpy or cupy array
        xp_orig = data
        if device.kind == "gpu":
            xp_round_tripped = cp.array(Tensor(xp_orig, device=device).op.data.cpu_view(cp.float32))
        else:
            xp_round_tripped = np.array(Tensor(xp_orig, device=device).op.data.cpu_view(np.float32))
        assert (xp_round_tripped == xp_orig).all()
        # assert xp_round_tripped.data == xp_orig.data

        # Assert round-tripping for Torch tensor
        torch_orig = torch.as_tensor(data)
        torch_round_tripped = torch.as_tensor(Tensor(torch_orig, device=device).op.data.cpu_view(np.float32))
        assert torch.equal(torch_round_tripped, torch_orig)
        # Below fails as we do allocate a new np array from Torch tensor data.
        # assert torch_data_round_tripped.data_ptr == torch_data.data_ptr

        # Assert round-tripping for Jax data
        if device.kind == "gpu":
            # TODO: Enable this when upgrade to 12.2.
            # Also, fix explicit .get() call here. can we construct Jax array from cupy directly.
            jax_orig = jax.device_put(jnp.array(data.get()), jax.devices("gpu")[0])
        else:
            jax_orig = jax.device_put(jnp.array(data), jax.devices("cpu")[0])
        jax_round_tripped = jnp.array(Tensor(jax_orig, device=device).op.data.cpu_view(np.float32))
        assert jnp.array_equal(jax_round_tripped, jax_orig)
        # Figure out how to compare two Jax data memory pointers.

        # Assert round-tripping for List data
        if device.kind == "cpu":
            list_orig = data.tolist()
            list_round_tripped = Tensor(list_orig, shape=(2,)).op.data.cpu_view(np.float32).tolist()
            assert list_round_tripped == list_orig
            # assert id(list_round_tripped) == id(list_orig)

    def test_tensor_round_tripping(self):
        from tripy.common.device import device as make_device

        self._test_round_tripping(np.ones(2, np.float32), device=make_device("cpu"))
        self._test_round_tripping(cp.ones(2, cp.float32), device=make_device("gpu"))
