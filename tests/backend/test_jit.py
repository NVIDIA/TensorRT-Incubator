import pytest
import tempfile
import os
import numpy as np
import cupy as cp

import tripy


@pytest.fixture
def init_tensors():
    a = tripy.Tensor(np.array([2, 3], dtype=np.float32), device=tripy.device("gpu"))
    b = tripy.Tensor(np.ones(2, dtype=np.float32), device=tripy.device("gpu"))
    return a, b


class TestJIT:
    def test_type_decorator(self):
        @tripy.jit
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        assert isinstance(func, tripy.jit)

    def test_type_decorator_kwargs(self):
        @tripy.jit(dummy=1)
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        assert isinstance(func, tripy.jit)

    def test_type_function(self):
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        jitted_func = tripy.jit(func)
        assert isinstance(jitted_func, tripy.jit)

    def test_functional_decorator(self, init_tensors):
        @tripy.jit
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

        jitted_func = tripy.jit(func)
        a, b = init_tensors
        c, d = jitted_func(a, b)
        assert (c.numpy() == np.array([3.0, 4.0], dtype=np.float32)).all() and (
            d.numpy() == np.array([6.0, 8.0], dtype=np.float32)
        ).all()

    def test_functional_decorator_kwargs(self, init_tensors):
        # kwargs are not used by jit implementation as of 11/14/2023.
        @tripy.jit(autotune=2)
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
        @tripy.jit(const_argnums=(0,))
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

        jitted_func = tripy.jit(
            func,
            const_argnums=(1,),
        )
        a, b = init_tensors
        c, d = jitted_func(a, b)
        assert (c.numpy() == np.array([3.0, 4.0], dtype=np.float32)).all() and (
            d.numpy() == np.array([6.0, 8.0], dtype=np.float32)
        ).all()

    def test_cache_decorator(self, init_tensors):
        @tripy.jit
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

    def test_cache_save_load(self, init_tensors):
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            jitted_func.save(tmp_dir)
            assert os.path.exists(tmp_dir) and len(os.listdir(tmp_dir)) == 1

            new_jitted_func = tripy.jit(
                func,
                const_argnums=(1,),
            )
            new_jitted_func.load(tmp_dir)
            # check the engine is loaded
            assert len(jitted_func.cache) == 1
            c, d = new_jitted_func(a, b)
            # check correctness of loaded engine
            assert (c.eval().view().tolist() == [3.0, 4.0]) and (d.eval().view().tolist() == [6.0, 8.0])
            # check the loaded engine is reused
            assert len(jitted_func.cache) == 1

    def test_cache_implicit_save_load(self, init_tensors):
        def func(a, b):
            c = a + b
            d = c + c
            return c, d

        with tempfile.TemporaryDirectory() as tmp_dir:
            jitted_func = tripy.jit(
                func,
                const_argnums=(1,),
                cache_dir=tmp_dir,
            )
            assert jitted_func.cache_dir == tmp_dir

            a, b = init_tensors
            c, d = jitted_func(a, b)

            jitted_func.save()
            # check cached engine is saved
            assert len(os.listdir(tmp_dir)) == 1

            new_jitted_func = tripy.jit(
                func,
                const_argnums=(1,),
                cache_dir=tmp_dir,
            )
            assert new_jitted_func.cache_dir == tmp_dir
            # check cached engine is loaded
            assert len(new_jitted_func.cache) == 1
