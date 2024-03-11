import numpy as np
import pytest
from unittest.mock import patch

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
        # Make sure that there is only one cached executable for the cache key.
        assert len(list(add.cache.values())[0]) == 1

    def test_print_warnings(self, capsys):
        random_data = np.random.rand(3).astype(np.float32)
        a = tp.Tensor(random_data, device=tp.device("gpu"))

        @tp.jit
        def add(a):
            print("Print in function jit mode.")
            return a

        _ = add(a).eval()
        _ = add(a).eval()

        captured = capsys.readouterr()
        assert "Usage of print statement in jitted functions is not recommended" in captured.out

    def test_print_warnings_nested_class(self, capsys):
        random_data = np.random.rand(3, 4).astype(np.float32)
        a = tp.Tensor(random_data, device=tp.device("gpu"))

        class Dummy(tp.Module):
            def __init__(self):
                super().__init__()

            def __call__(self, x):
                print("Dummy call")
                return x

        class Network(tp.Module):
            def __init__(self):
                super().__init__()
                self.linear = tp.Linear(4, 2)
                self.dummy = Dummy()

            def __call__(self, x):
                return self.linear(self.dummy(x))

        net = Network()
        net = tp.jit(net)

        # Call eval twice to show print only gets triggered once during tracing.
        _ = net(a).eval()
        _ = net(a).eval()

        captured = capsys.readouterr()

        assert "Usage of print statement in jitted functions is not recommended" in captured.out
        # Verify that warning logs show the nested print statment.
        assert "'Dummy' : print(\"Dummy call\")" in captured.out

    def test_jit_warn_illegal_behavior(self, capsys):

        random_data = np.random.rand(3).astype(np.float32)
        a = tp.Tensor(random_data, device=tp.device("gpu"))

        with patch("pdb.set_trace"):

            @tp.jit
            def add(a):
                t = tp.Tensor([1.0, 2.0, 3.0], shape=(tp.Dim(3, 2, 3, 4),))
                out = a + t
                import pdb

                pdb.set_trace()
                return out

            _ = add(a).eval()

        captured = capsys.readouterr()
        assert "Initializing dynamic shape tensor in jitted functions is not recommended" in captured.out
        assert "Using pdb inside jitted function is not recommended" in captured.out

    def test_jit_in_jit(self):
        # Verify that jit function called in another jit function does not cause inner jit function args to be evaluated.

        @tp.jit
        def inner(a, b):
            return a * b

        @tp.jit
        def outer(a, b):
            a = a - 2.0
            b = b + 2.0
            out = inner(a, b)
            out = out / 3.0
            return out

        a = tp.ones((2, 3))
        a.name = "c"
        b = tp.ones((2, 3))
        b.name = "b"

        c = outer(a, b).eval()

        # mul cache keys will be empty since jit wasn't used and the function body was traced by the parent jit function.
        assert len(inner.cache.keys()) == 0
        assert len(outer.cache.keys()) == 1

    @pytest.mark.parametrize(
        "use_jit_args",
        [True, False],
    )
    def test_jit_in_jit_class(self, use_jit_args):
        # Verify that jit class called in another jit class does not cause inner jit function args to be evaluated.

        # A wrapper function for the __call__ method without jit args
        @tp.jit
        def call_without_jit(self, a, b):
            return a * b

        class Inner(tp.Module):
            def __init__(self):
                super().__init__()

            @tp.jit(dummy=1)
            def __call__(self, a, b):
                return a * b

        if not use_jit_args:
            # Directly replace the __call__ method for this test case
            Inner.__call__ = call_without_jit

        class Outer(tp.Module):
            def __init__(self):
                super().__init__()
                self.mul = Inner()

            def __call__(self, a, b):
                a = a - 2.0
                b = b + 2.0
                out = self.mul(a, b)
                out = out / 3.0
                return out

        net = Outer()
        net = tp.jit(net)

        a = tp.ones((2, 3))
        a.name = "c"
        b = tp.ones((2, 3))
        b.name = "b"

        c = net(a, b).eval()
        assert len(Inner.__call__.cache.keys()) == 0
