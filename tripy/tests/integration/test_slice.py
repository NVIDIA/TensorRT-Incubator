#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cupy as cp
import numpy as np
import math
import pytest

import nvtripy as tp


class TestSliceOp:
    @pytest.mark.parametrize(
        "dims_a, slice_func",
        [
            ((2,), lambda t: t[-1]),
            ((4,), lambda t: t[-2]),
            ((4,), lambda t: t[1:]),
            ((2, 3, 4), lambda t: t[1, 2, 3]),
            # flip one dimension
            ((2, 3, 4), lambda t: t[:, ::-1, :]),
            # negative step size that evenly and unevenly divides
            ((2, 3, 4), lambda t: t[:, :, ::-2]),
            ((2, 3, 4), lambda t: t[:, :, ::-3]),
            # both bounds given with negative step size
            ((10,), lambda t: t[8:2:-2]),
            # one bound given with negative step size
            ((10,), lambda t: t[8::-2]),
            ((10,), lambda t: t[:2:-2]),
            # both bounds with uneven step size
            ((10,), lambda t: t[8:2:-3]),
            # not the same thing as [10::-1] -- this one leaves off the last element
            ((10,), lambda t: t[10:0:-1]),
            # clamps the start index for negative step size
            ((10,), lambda t: t[1024:0:-1]),
            ((1, 2, 1, 4), lambda t: t[:, 1, 0, 2:-1]),
            # ensure that if a slice upper bound is past the end, it is clamped
            ((2, 3, 4), lambda t: t[:3, :4, :5]),
            # TODO #156: implement when infer_rank is available on frontend tensor
            # The current way to dynamically add start,limit,stride content to slice params is very hacky and not worth adding right now.
            # ((2,3,4,5), lambda t: t[:1]),
            # empty dimension
            ((2, 3, 4), lambda t: t[0:0, :-2, 1:]),
            # out of bounds slice
            ((2, 3, 4), lambda t: t[3:5, :-2, 1:]),
            # ensure that absurdly out of bounds dims will result in an empty tensor
            ((10,), lambda t: t[1234:5678]),
            # also check with negative step size
            ((10,), lambda t: t[1234:5678:-1]),
            # also in the usual index ordering
            ((10,), lambda t: t[5678:1234:-1]),
            # both out of bounds and negative (should be an empty tensor)
            ((5,), lambda t: t[-5:-12]),
            # also with negative step size
            ((5,), lambda t: t[-12:-5:-1]),
        ],
    )
    def test_static_slice_op(self, dims_a, slice_func, eager_or_compiled):
        a_cp = cp.arange(np.prod(dims_a)).reshape(dims_a).astype(np.float32)
        a = tp.Tensor(a_cp, device=tp.device("gpu"))

        def func(a):
            return slice_func(a)

        out = eager_or_compiled(func, a)
        assert np.array_equal(cp.from_dlpack(out).get(), slice_func(a_cp).get())

    def test_slice_as_gather(self, eager_or_compiled):
        x_data = [0, 1, 2]
        y_data = [3, 4, 5]
        x = tp.Tensor(x_data)
        y = tp.Tensor(y_data)

        def slice(y, x):
            return y[x]

        output = eager_or_compiled(slice, y, x)

        x_cp = cp.array(x_data)
        y_cp = cp.array(y_data)

        assert np.array_equal(cp.from_dlpack(output).get(), y_cp[x_cp].get())

        x_shape = (2, 2)
        y_shape = (4, 3, 2)
        x_vol = math.prod(x_shape)
        y_vol = math.prod(y_shape)
        x = tp.reshape(tp.arange(x_vol, dtype=tp.int32), x_shape)
        y = tp.reshape(tp.arange(y_vol), y_shape)
        output = eager_or_compiled(slice, y, x)

        x_cp = cp.arange(x_vol, dtype=cp.int32).reshape(x_shape)
        y_cp = cp.arange(y_vol).reshape(y_shape)

        assert np.array_equal(cp.from_dlpack(output).get(), y_cp[x_cp].get())

    def test_scalar_index(self):
        a = tp.ones((2, 3, 4))
        assert a[0].shape == [3, 4]
        assert a[0:1].shape == [1, 3, 4]

    def test_end_clamping(self):
        a = tp.ones((2, 3, 4))
        # equivalent to a[0:2, 0:3, 3:4]
        assert a[0:7, 0:12, 3:5].shape == [2, 3, 1]

    def test_tensor_index(self):
        idx = tp.Tensor(1, dtype=tp.int32)
        a = tp.ones((2, 3))
        b = a[idx]
        assert b.shape == [3]

    def test_empty_slice(self):
        a = tp.ones((2, 3, 4))
        b = a[3:2:1]
        assert b.shape == [0, 3, 4]
        assert cp.from_dlpack(b).get().tolist() == []
