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

from tests.helper import torch_type_supported, NUMPY_TYPES


class TestFunctional:
    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_add_two_tensors(self, kind):
        arr = np.array([2, 3], dtype=np.float32)
        a = Tensor(arr, device=device(kind))
        b = Tensor(np.ones(2, dtype=np.float32), device=device(kind))

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
