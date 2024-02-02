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

    def test_dynamic_shapes(self):
        random_data = np.random.rand(3).astype(np.float32)
        dynamic_dim = tp.Dim(3, min=2, opt=3, max=10)

        a = tp.Tensor(random_data, shape=(dynamic_dim,), device=tp.device("gpu"))
        b = tp.Tensor(random_data, shape=(dynamic_dim,), device=tp.device("gpu"))

        @tp.jit
        def add(a, b):
            return a + b

        # Compile once with dynamic shapes
        out = add(a, b)
        assert np.array_equal(out.numpy(), random_data + random_data)

        # We should be able to use other shapes without recompiling
        a = tp.ones((6,))
        assert np.array_equal(add(a, a).numpy(), np.ones((6,), dtype=np.float32) + np.ones((6,), dtype=np.float32))
        assert len(add.cache) == 1