import pytest
import numpy as np

import tripy
from tripy.util.util import all_same


@pytest.fixture
def init_tensors():
    a = tripy.Tensor([2.0, 3.0], shape=(2,), device=tripy.device("gpu"))
    # (25): Add tests for broadcasting.
    b = tripy.Tensor([1.0, 1.0], shape=(2,), device=tripy.device("gpu"))
    return a, b


class TestJIT:
    def test_functional_decorator(self, init_tensors):
        @tripy.jit
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        a, b = init_tensors
        c, d = func(a, b)
        assert all_same(c.eval(), [3.0, 4.0]) and all_same(d.eval(), [6.0, 8.0])

    def test_functional_function(self, init_tensors):
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = tripy.jit(func)
        a, b = init_tensors
        c, d = jitted_func(a, b)
        assert all_same(c.eval(), [3.0, 4.0]) and all_same(d.eval(), [6.0, 8.0])

    def test_functional_decorator_kwargs(self, init_tensors):
        # kwargs are not used by jit implementation as of 11/14/2023.
        @tripy.jit(autotune=2)
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        a, b = init_tensors
        c, d = func(a, b)
        assert all_same(c.eval(), [3.0, 4.0]) and all_same(d.eval(), [6.0, 8.0])

    def test_functional_decorator_const_argnums(self, init_tensors):
        @tripy.jit(const_argnums=(0,))
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        a, b = init_tensors
        c, d = func(a, b)
        assert all_same(c.eval(), [3.0, 4.0]) and all_same(d.eval(), [6.0, 8.0])

    def test_functional_function_const_argnums(self, init_tensors):
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = tripy.jit(
            func,
            const_argnums=(1,),
        )
        a, b = init_tensors
        c, d = jitted_func(a, b)
        assert all_same(c.eval(), [3.0, 4.0]) and all_same(d.eval(), [6.0, 8.0])

    def test_cache_decorator(self, init_tensors):
        @tripy.jit
        def func(a, b, option=None):
            c = a + b
            d = c + c
            return c, d

        a, b = init_tensors
        c, d = func(a, b)
        assert len(func.cache) == 1

        a = a + a
        b = b + b
        # Check cached executable is reused when different inputs have same shapes
        c, d = func(a, b)
        assert len(func.cache) == 1

        # Different kwargs will trigger recompilation
        c, d = func(a, b, option=True)
        assert len(func.cache) == 2

    def test_cache_function(self, init_tensors):
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = tripy.jit(
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
