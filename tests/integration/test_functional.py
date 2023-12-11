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

from tests.helper import torch_type_supported
from tests.helper import NUMPY_TYPES


@pytest.fixture
def a():
    return [2.0, 3.0]


@pytest.fixture
def b():
    return [1.0, 1.0]


@pytest.fixture
def init_tensors(a, b):
    def init_list(z):
        data = [np.array(z, dtype=dtype) for dtype in NUMPY_TYPES]
        l = []
        l.append(
            [np.array(z, dtype=np.float32).tolist(), np.array(z, dtype=np.int32).tolist()]
        )  # Only float and int are supported for a list.
        l.append(data)
        # Extend the data list for Cupy arrays
        l.append([cp.array(d) for d in data])
        # Extend the data list for Torch CPU tensors
        l.append([torch.tensor(d) for d in list(filter(torch_type_supported, data))])
        # Extend the data list for Torch GPU tensors
        l.append([torch.tensor(d).to(torch.device("cuda")) for d in list(filter(torch_type_supported, data))])
        # Extend the data list for Jax CPU arrays
        l.append([jax.device_put(jnp.array(d), jax.devices("cpu")[0]) for d in data])
        # Extend the data list for Jax GPU arrays
        l.append([jax.device_put(jnp.array(d), jax.devices("cuda")[0]) for d in data])
        return l

    al = init_list(a)
    bl = init_list(b)
    result = [{"a": x, "b": y} for x, y in zip(al, bl)]
    return result


class TestFunctional:
    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_add_two_tensors(self, kind, init_tensors):
        for data in init_tensors:
            for a_, b_ in zip(data["a"], data["b"]):
                a = Tensor(a_, shape=(2,), device=device(kind))
                b = Tensor(b_, shape=(2,), device=device(kind))
                c = a + b
                out = c + c
                assert all_same(out.eval(), [6.0, 8.0])

    @pytest.mark.parametrize("dim", [Dim(2, min=2, opt=2, max=2)])
    def test_add_two_tensors_dynamic(self, dim):
        a = Tensor([1.0, 1.0], shape=(dim))
        b = Tensor([1.0, 1.0], shape=(dim))

        @jit
        def func(a, b):
            c = a + b
            return c

        out = func(a, b)
        assert all_same(out.eval(), [2.0, 2.0])

    def test_multi_output_flat_ir(self):
        a = Tensor([1.0, 1.0], shape=(2,))
        b = Tensor([1.0, 1.0], shape=(2,))
        c = a + b
        d = c + c
        flat_ir = FlatIR([c, d])

        compiler = FlatIRCompiler()
        with FlatIRExecutor(compiler.compile(flat_ir)) as executor:
            out = executor.execute()
            assert len(out) == 2 and all_same(out[0], [2.0, 2.0]) and all_same(out[1], [4.0, 4.0])
