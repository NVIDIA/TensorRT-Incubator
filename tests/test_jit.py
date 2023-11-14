import numpy as np

from tripy.frontend import Tensor
from tripy.flat_ir import FlatIR
from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor
from tripy.jit import JIT as jit


class TestJIT:
    def test_jit_decorator(self):
        a = Tensor(np.array([2, 3], dtype=np.float32))
        b = Tensor(np.ones(2, dtype=np.float32))

        @jit
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        c, d = func(a, b)
        assert len(func.cache) == 1

        # Check that cached implementation is used and cache does not grow.
        c, d = func(a, b)
        assert len(func.cache) == 1

        assert (c.eval() == np.array([3.0, 4.0])).all() and (d.eval() == np.array([6.0, 8.0])).all()

        a = a + a
        b = b + b
        # Check that cache grows since input has changed.
        # When jit/trt-mlir handles input tensors, this test case will need to change shape of a instead of data.
        c, d = func(a, b)
        assert len(func.cache) == 2

    def test_jit_function(self):
        a = Tensor(np.array([2, 3], dtype=np.float32))
        b = Tensor(np.ones(2, dtype=np.float32))

        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = jit(func)
        c, d = jitted_func(a, b)
        assert (c.eval() == np.array([3.0, 4.0])).all() and (d.eval() == np.array([6.0, 8.0])).all()

    def test_jit_decorator_kwargs(self):
        a = Tensor(np.array([2, 3], dtype=np.float32))
        b = Tensor(np.ones(2, dtype=np.float32))

        # kwargs are not used by jit implementation as of 11/14/2023.
        @jit(autotune=2)
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        c, d = func(a, b)
        assert (c.eval() == np.array([3.0, 4.0])).all() and (d.eval() == np.array([6.0, 8.0])).all()
