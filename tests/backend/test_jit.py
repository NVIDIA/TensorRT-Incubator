import pytest
import numpy as np

import tripy


class TestJIT:
    def test_jit_decorator(self):
        a = tripy.Tensor(np.array([2, 3], dtype=np.float32))
        b = tripy.Tensor(np.ones(2, dtype=np.float32))

        @tripy.jit
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
        a = tripy.Tensor(np.array([2, 3], dtype=np.float32))
        b = tripy.Tensor(np.ones(2, dtype=np.float32))

        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = tripy.jit(func)
        c, d = jitted_func(a, b)
        assert (c.eval() == np.array([3.0, 4.0])).all() and (d.eval() == np.array([6.0, 8.0])).all()

    def test_jit_decorator_kwargs(self):
        a = tripy.Tensor(np.array([2, 3], dtype=np.float32))
        b = tripy.Tensor(np.ones(2, dtype=np.float32))

        # kwargs are not used by jit implementation as of 11/14/2023.
        @tripy.jit(autotune=2)
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        c, d = func(a, b)
        assert (c.eval() == np.array([3.0, 4.0])).all() and (d.eval() == np.array([6.0, 8.0])).all()

    def test_jit_decorator_const_argnums(self):
        a = tripy.Tensor(np.array([2, 3], dtype=np.float32), device=tripy.device("gpu"))
        b = tripy.Tensor(np.ones(2, dtype=np.float32), device=tripy.device("gpu"))

        @tripy.jit(const_argnums=(0,))
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        c, d = func(a, b)
        assert (c.eval() == np.array([3.0, 4.0])).all() and (d.eval() == np.array([6.0, 8.0])).all()

    def test_jit_function_const_argnums(self):
        a = tripy.Tensor(np.array([2, 3]), device=tripy.device("gpu"))
        b = tripy.Tensor(np.ones(2), device=tripy.device("gpu"))

        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = tripy.jit(
            func,
            const_argnums=(0, 1),
        )
        c, d = jitted_func(a, b)
        assert (c.eval() == np.array([3.0, 4.0])).all() and (d.eval() == np.array([6.0, 8.0])).all()
