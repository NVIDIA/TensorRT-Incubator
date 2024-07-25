import cupy as cp
import pytest

import tripy as tp
from tests import helper


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def returns_non_tensor(a):
    return "not a tensor"


def returns_nothing(a):
    return


def returns_multiple_tensors(a, b):
    return a + b, a - b


def variadic_positional(*args):
    pass


def variadic_keyword(**kwargs):
    pass


class TestInput:
    @pytest.mark.parametrize(
        "shape, expected_min, expected_opt, expected_max",
        [
            # min/opt/max explicitly specified
            ([(1, 2, 3)], (1,), (2,), (3,)),
            # Only one value specified
            ([1], (1,), (1,), (1,)),
        ],
    )
    def test_shapes_normalized(self, shape, expected_min, expected_opt, expected_max):
        inp = tp.InputInfo(shape=shape, dtype=tp.float32)

        assert inp.shape_bounds.min == expected_min
        assert inp.shape_bounds.opt == expected_opt
        assert inp.shape_bounds.max == expected_max

    @pytest.mark.parametrize(
        "shape, expected_error",
        [
            # Not a number
            (
                (tp.int32, 1),
                "Shape values should be either a single number or a Tuple specifying min/opt/max bounds.",
            ),
            # Too few elements in dimension
            (((1, 1), 1), "Incorrect number of shape values provided"),
            # Too many elements in dimension
            (((1, 1, 1, 1), 1), "Incorrect number of shape values provided"),
            # Tuple containing a non-number
            (((tp.int32, 1, 1), 1), "Shape values must be numbers"),
        ],
    )
    def test_invalid_shape(self, shape, expected_error):
        with helper.raises(tp.TripyException, expected_error):
            tp.InputInfo(shape=shape, dtype=tp.float32)


class TestCompile:
    # TODO (#246): Verify that it's actually compiling somehow here and below.
    # Need to return something programatically queriable from compile to do this.
    def test_function(self):
        compiled_gelu = tp.Compiler(tp.relu).compile(tp.InputInfo((2, 2), dtype=tp.float32))

        inp = tp.ones((2, 2), dtype=tp.float32)
        out = compiled_gelu(inp)

        # TODO (#225): Replace with tp.all
        assert cp.array_equal(cp.from_dlpack(out), cp.from_dlpack(tp.relu(inp)))

    def test_module(self):
        layernorm = tp.LayerNorm(2)
        compiled_layernorm = tp.Compiler(layernorm).compile(tp.InputInfo((2, 2), dtype=tp.float32))

        inp = tp.ones((2, 2), dtype=tp.float32)
        out = compiled_layernorm(inp)

        assert cp.array_equal(cp.from_dlpack(out), cp.from_dlpack(layernorm(inp)))

    def test_compile_arg_order_irrelevant(self):
        compiler = tp.Compiler(sub)

        # The order of arguments we specify to `compile` should not affect the order
        # of the arguments in the compiled function, which should just follow the order
        # of the original function.
        compiled_sub = compiler.compile(
            b=tp.InputInfo((2, 2), dtype=tp.float32), a=tp.InputInfo((2, 2), dtype=tp.float32)
        )

        a = tp.ones((2, 2), dtype=tp.float32) * 2
        b = tp.ones((2, 2), dtype=tp.float32)

        # Compiled function should still take arguments in (a, b) order.
        out = compiled_sub(a, b)
        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 2), dtype=cp.float32))

    @pytest.mark.parametrize("b", [2, tp.ones((2, 2), dtype=tp.float32) * 2])
    def test_constants_baked(self, b):
        compiler = tp.Compiler(add)

        # Any non-InputInfo argument to compile is baked into the compiled function.
        compiled_add = compiler.compile(tp.InputInfo((2, 2), dtype=tp.float32), b)

        a = tp.zeros((2, 2), dtype=tp.float32)

        out = compiled_add(a)

        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 2), dtype=cp.float32) * 2)

    @pytest.mark.parametrize("func", [variadic_positional, variadic_keyword])
    def test_variadic_arguments_rejected(self, func):
        with helper.raises(tp.TripyException, "Variadic positional/keyword arguments are not currently supported."):
            tp.Compiler(func)

    @pytest.mark.parametrize("func", [returns_non_tensor, returns_nothing])
    def test_invalid_return_rejected(self, func):
        compiler = tp.Compiler(func)
        with helper.raises(tp.TripyException, "Function must return 1 or more Tensors"):
            compiler.compile(tp.InputInfo((2, 2), dtype=tp.float32))

    def test_multiple_return_values(self):
        compiler = tp.Compiler(returns_multiple_tensors)
        compiled_func = compiler.compile(tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32))

        a = tp.ones((2, 2), dtype=tp.float32) * 2
        b = tp.ones((2, 2), dtype=tp.float32)

        plus, minus = compiled_func(a, b)

        assert cp.array_equal(cp.from_dlpack(plus), cp.ones((2, 2), dtype=cp.float32) * 3)
        assert cp.array_equal(cp.from_dlpack(minus), cp.ones((2, 2), dtype=cp.float32))

    def test_incorrect_dtype_rejected(self):
        compiler = tp.Compiler(add)
        a = tp.ones((2, 2), dtype=tp.int32)

        with helper.raises(tp.TripyException, "Unexpected tensor data type.", has_stack_info_for=[a]):
            compiled_add = compiler.compile(
                tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32)
            )
            compiled_add(a, a)

    @pytest.mark.skip("TODO (#155): Re-enable once we no longer implicitly copy inputs to device")
    def test_incorrect_device_rejected(self):
        compiler = tp.Compiler(add)
        compiled_add = compiler.compile(tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32))
        a = tp.copy(tp.ones((2, 2), dtype=tp.float32), device=tp.device("cpu"))

        with helper.raises(tp.TripyException):
            compiled_add(a, a)

    # TODO (#244): Add multi-profile test
    def test_dynamic_shapes(self):
        compiler = tp.Compiler(add)

        compiled_add = compiler.compile(
            tp.InputInfo(((1, 2, 3), 1), dtype=tp.float32), tp.InputInfo(((1, 2, 3), 1), dtype=tp.float32)
        )

        out = compiled_add(tp.ones((2, 1), dtype=tp.float32), tp.ones((2, 1), dtype=tp.float32))
        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 1), dtype=cp.float32) * 2)

        out = compiled_add(tp.ones((3, 1), dtype=tp.float32), tp.ones((3, 1), dtype=tp.float32))
        assert cp.array_equal(cp.from_dlpack(out), cp.ones((3, 1), dtype=cp.float32) * 2)


# TODO (#256): Remove these tests and replace with exhaustive integration testing
class TestCompiledOps:
    def test_cast(self):
        compiler = tp.Compiler(tp.cast)
        compiled_cast = compiler.compile(tp.InputInfo((2, 2), dtype=tp.float32), dtype=tp.int32)

        a = tp.ones((2, 2), dtype=tp.float32)
        out = compiled_cast(a)

        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 2), dtype=cp.int32))

    def test_linear(self):
        linear = tp.Linear(2, 3)
        compiler = tp.Compiler(linear)

        compiled_linear = compiler.compile(tp.InputInfo((2, 2), dtype=tp.float32))

        a = tp.ones((2, 2), dtype=tp.float32)

        out = compiled_linear(a)

        assert cp.array_equal(cp.from_dlpack(out), cp.from_dlpack(linear(a)))
