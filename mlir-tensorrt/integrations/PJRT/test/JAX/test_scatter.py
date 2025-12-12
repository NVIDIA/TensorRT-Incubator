# RUN: %pick-one-gpu %mlir-trt-jax-py %s

"""Scatter-related tests reproduced from JAX's test/lax_test.py.

This test contains just a sampling of scatter-related tests.
They have been modified to run all test cases rather than just JAX_NUM_GENERATED_CASES
in order to fully test the scatter codegen functionality.

Original code link:
https://github.com/jax-ml/jax/blob/main/tests/lax_test.py

Original license:
Copyright 2018 The JAX Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
"""

import pytest
import numpy as np
import jax.lax as lax
from functools import partial
from jax._src import test_util as jtu
from absl.testing import parameterized
from absl.testing import absltest
import jax.numpy as jnp


float_dtypes = jtu.dtypes.all_floating
inexact_dtypes = jtu.dtypes.all_inexact


class ScatterTest(jtu.JaxTestCase):
    """Tests for jax.lax.scatter"""

    @parameterized.product(
        [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape, dnums=dnums)
            for arg_shape, idxs, update_shape, dnums in [
                (
                    (5,),
                    np.array([[0], [2]]),
                    (2,),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10,),
                    np.array([[0], [0], [0]]),
                    (3, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10, 5),
                    np.array([[0], [2], [1]]),
                    (3, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (2, 5),
                    np.array([[[0], [2]], [[1], [1]]]),
                    (2, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(1,),
                        scatter_dims_to_operand_dims=(1,),
                        operand_batching_dims=(0,),
                        scatter_indices_batching_dims=(0,),
                    ),
                ),
                (
                    (2, 3, 10),
                    np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
                    (3, 2, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(2,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(2,),
                        operand_batching_dims=(0, 1),
                        scatter_indices_batching_dims=(1, 0),
                    ),
                ),
            ]
        ],
        dtype=inexact_dtypes,
        mode=["clip", "fill", None],
        op=[lax.scatter_add, lax.scatter_sub],
    )
    def testScatterAddSub(self, arg_shape, dtype, idxs, update_shape, dnums, mode, op):
        rng = jtu.rand_default(self.rng())
        rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
        rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
        args_maker = lambda: [
            rng(arg_shape, dtype),
            rand_idxs(),
            rng(update_shape, dtype),
        ]
        fun = partial(op, dimension_numbers=dnums, mode=mode)
        self._CompileAndCheck(fun, args_maker)

    @parameterized.product(
        [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape, dnums=dnums)
            for arg_shape, idxs, update_shape, dnums in [
                (
                    (5,),
                    np.array([[0], [2]]),
                    (2,),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10,),
                    np.array([[0], [0], [0]]),
                    (3, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10, 5),
                    np.array([[0], [2], [1]], dtype=np.uint64),
                    (3, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (2, 5),
                    np.array([[[0], [2]], [[1], [1]]]),
                    (2, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(1,),
                        scatter_dims_to_operand_dims=(1,),
                        operand_batching_dims=(0,),
                        scatter_indices_batching_dims=(0,),
                    ),
                ),
                (
                    (2, 3, 10),
                    np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
                    (3, 2, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(2,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(2,),
                        operand_batching_dims=(0, 1),
                        scatter_indices_batching_dims=(1, 0),
                    ),
                ),
            ]
        ],
        dtype=float_dtypes,
    )
    def testScatterMin(self, arg_shape, dtype, idxs, update_shape, dnums):
        rng = jtu.rand_default(self.rng())
        rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
        rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
        args_maker = lambda: [
            rng(arg_shape, dtype),
            rand_idxs(),
            rng(update_shape, dtype),
        ]
        fun = partial(lax.scatter_min, dimension_numbers=dnums)
        self._CompileAndCheck(fun, args_maker)

    @parameterized.product(
        [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape, dnums=dnums)
            for arg_shape, idxs, update_shape, dnums in [
                (
                    (5,),
                    np.array([[0], [2]]),
                    (2,),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10,),
                    np.array([[0], [0], [0]]),
                    (3, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10, 5),
                    np.array([[0], [2], [1]]),
                    (3, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (2, 5),
                    np.array([[[0], [2]], [[1], [1]]]),
                    (2, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(1,),
                        scatter_dims_to_operand_dims=(1,),
                        operand_batching_dims=(0,),
                        scatter_indices_batching_dims=(0,),
                    ),
                ),
                (
                    (2, 3, 10),
                    np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
                    (3, 2, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(2,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(2,),
                        operand_batching_dims=(0, 1),
                        scatter_indices_batching_dims=(1, 0),
                    ),
                ),
            ]
        ],
        dtype=float_dtypes,
    )
    def testScatterMax(self, arg_shape, dtype, idxs, update_shape, dnums):
        rng = jtu.rand_default(self.rng())
        rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
        rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
        args_maker = lambda: [
            rng(arg_shape, dtype),
            rand_idxs(),
            rng(update_shape, dtype),
        ]
        fun = partial(lax.scatter_max, dimension_numbers=dnums)
        self._CompileAndCheck(fun, args_maker)

    @parameterized.product(
        [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape, dnums=dnums)
            for arg_shape, idxs, update_shape, dnums in [
                (
                    (5,),
                    np.array([[0], [2]]),
                    (2,),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10,),
                    np.array([[0], [0], [0]]),
                    (3, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10, 5),
                    np.array([[0], [2], [1]]),
                    (3, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (2, 5),
                    np.array([[[0], [2]], [[1], [1]]]),
                    (2, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(1,),
                        scatter_dims_to_operand_dims=(1,),
                        operand_batching_dims=(0,),
                        scatter_indices_batching_dims=(0,),
                    ),
                ),
                (
                    (2, 3, 10),
                    np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
                    (3, 2, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(2,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(2,),
                        operand_batching_dims=(0, 1),
                        scatter_indices_batching_dims=(1, 0),
                    ),
                ),
            ]
        ],
        dtype=float_dtypes,
    )
    def testScatterApply(self, arg_shape, dtype, idxs, update_shape, dnums):
        rng = jtu.rand_default(self.rng())
        rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
        rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
        args_maker = lambda: [rng(arg_shape, dtype), rand_idxs()]
        fun = partial(
            lax.scatter_apply,
            func=jnp.sin,
            update_shape=update_shape,
            dimension_numbers=dnums,
        )
        self._CompileAndCheck(fun, args_maker)

    @parameterized.product(
        [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape, dnums=dnums)
            for arg_shape, idxs, update_shape, dnums in [
                (
                    (5,),
                    np.array([[0], [2]]),
                    (2,),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10,),
                    np.array([[0], [0], [0]]),
                    (3, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10, 5),
                    np.array([[0], [2], [1]]),
                    (3, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (2, 5),
                    np.array([[[0], [2]], [[1], [1]]]),
                    (2, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(1,),
                        scatter_dims_to_operand_dims=(1,),
                        operand_batching_dims=(0,),
                        scatter_indices_batching_dims=(0,),
                    ),
                ),
                (
                    (2, 3, 10),
                    np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
                    (3, 2, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(2,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(2,),
                        operand_batching_dims=(0, 1),
                        scatter_indices_batching_dims=(1, 0),
                    ),
                ),
            ]
        ],
        dtype=float_dtypes,
    )
    def testScatter(self, arg_shape, dtype, idxs, update_shape, dnums):
        rng = jtu.rand_default(self.rng())
        rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
        rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
        args_maker = lambda: [
            rng(arg_shape, dtype),
            rand_idxs(),
            rng(update_shape, dtype),
        ]
        fun = partial(lax.scatter, dimension_numbers=dnums)
        self._CompileAndCheck(fun, args_maker)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
