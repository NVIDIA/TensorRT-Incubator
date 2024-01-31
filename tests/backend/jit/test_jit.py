import os
import tempfile

import numpy as np
import pytest

import tripy as tp


@pytest.fixture
def init_tensors():
    a = tp.Tensor(np.array([2, 3], dtype=np.float32), device=tp.device("gpu"))
    b = tp.Tensor(np.ones(2, dtype=np.float32), device=tp.device("gpu"))
    return a, b


class TestJIT:
    def test_type_decorator(self):
        @tp.jit
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        assert isinstance(func, tp.jit)

    def test_type_decorator_kwargs(self):
        @tp.jit(dummy=1)
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        assert isinstance(func, tp.jit)

    def test_type_function(self):
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = tp.jit(func)
        assert isinstance(jitted_func, tp.jit)

    def test_functional_decorator(self, init_tensors):
        @tp.jit
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        a, b = init_tensors
        c, d = func(a, b)
        assert (c.numpy() == np.array([3.0, 4.0], dtype=np.float32)).all() and (
            d.numpy() == np.array([6.0, 8.0], dtype=np.float32)
        ).all()

    def test_functional_function(self, init_tensors):
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = tp.jit(func)
        a, b = init_tensors
        c, d = jitted_func(a, b)
        assert (c.numpy() == np.array([3.0, 4.0], dtype=np.float32)).all() and (
            d.numpy() == np.array([6.0, 8.0], dtype=np.float32)
        ).all()

    def test_functional_decorator_kwargs(self, init_tensors):
        # kwargs are not used by jit implementation as of 11/14/2023.
        @tp.jit(autotune=2)
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        a, b = init_tensors
        c, d = func(a, b)
        assert (c.numpy() == np.array([3.0, 4.0], dtype=np.float32)).all() and (
            d.numpy() == np.array([6.0, 8.0], dtype=np.float32)
        ).all()

    def test_functional_decorator_const_argnums(self, init_tensors):
        @tp.jit(const_argnums=(0,))
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        a, b = init_tensors
        c, d = func(a, b)
        assert (c.numpy() == np.array([3.0, 4.0], dtype=np.float32)).all() and (
            d.numpy() == np.array([6.0, 8.0], dtype=np.float32)
        ).all()

    def test_functional_function_const_argnums(self, init_tensors):
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = tp.jit(
            func,
            const_argnums=(1,),
        )
        a, b = init_tensors
        c, d = jitted_func(a, b)
        assert (c.numpy() == np.array([3.0, 4.0], dtype=np.float32)).all() and (
            d.numpy() == np.array([6.0, 8.0], dtype=np.float32)
        ).all()

    def test_functional_io_order(self, init_tensors):
        @tp.jit
        def func(a, b):
            return b, a

        a, b = init_tensors
        c, d = func(a, b)
        assert (c.numpy() == b.numpy()).all() and (d.numpy() == a.numpy()).all()

    def test_cache_decorator(self, init_tensors):
        @tp.jit
        def func(a, b, option=False):
            c = a + b
            d = c + c
            if option is True:
                d = d + d
            return c, d

        a, b = init_tensors
        c, d = func(a, b)
        assert len(func.cache) == 1

        a = a + a
        b = b + b
        # Check cached executable is reused when function ir is the same
        c, d = func(a, b)
        assert len(func.cache) == 1

        # Different function ir will trigger recompilation
        c, d = func(a, b, option=True)
        assert len(func.cache) == 2

    def test_cache_function(self, init_tensors):
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = tp.jit(
            func,
            const_argnums=(1,),
        )
        a, b = init_tensors
        c, d = jitted_func(a, b)
        assert len(jitted_func.cache) == 1

        # change input tensor does not need recompilation
        a = a + a
        c, d = jitted_func(a, b)
        assert len(jitted_func.cache) == 1

        # different const tensor will trigger recompilation
        b = b + b
        c, d = jitted_func(a, b)
        assert len(jitted_func.cache) == 2
