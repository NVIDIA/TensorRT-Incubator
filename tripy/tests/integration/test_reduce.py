#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest
import torch

import tripy as tp


class TestReduceOp:
    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), (1, 2), True),
            ((2, 3), 1, True),
            ((2, 3, 4), (1, 2), False),
            ((2, 3, 4), None, False),
            ((2, 3, 4), None, True),
            ((2, 3, 4, 5), (-2, -1), True),
        ],
    )
    def test_all(self, x_shape, axis, keepdim, compile_fixture):
        x = np.array([i % 2 == 0 for i in np.arange(np.prod(x_shape))]).reshape(x_shape)
        a = tp.Tensor(x)
        out = compile_fixture(tp.all, a, dim=axis, keepdim=keepdim)
        expected = tp.Tensor(np.array(x.all(axis=axis, keepdims=keepdim)))
        # np.array is necessary to deal with case where x.all returns a numpy scalar (5th case)
        assert out.shape == expected.shape
        assert tp.allclose(out, expected)

    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), (1, 2), True),
            ((2, 3), 1, False),
            ((2, 3, 4), (1, 2), False),
            ((2, 3, 4), None, False),
            ((2, 3, 4), None, True),
            ((2, 3, 4, 5), (-2, -1), True),
        ],
    )
    def test_any(self, x_shape, axis, keepdim, compile_fixture):
        x = np.array([i % 2 == 0 for i in np.arange(np.prod(x_shape))]).reshape(x_shape)
        a = tp.Tensor(x)
        out = compile_fixture(tp.any, a, dim=axis, keepdim=keepdim)
        assert tp.allclose(out, tp.Tensor(np.array(x.any(axis=axis, keepdims=keepdim))))

    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            pytest.param(
                (2, 3, 4),
                (1, 2),
                True,
                marks=pytest.mark.skip(reason="For this test case without out.eval() tp.allclose fails. (Issue #)"),
            ),
            ((2, 3), 1, False),
            ((2, 3, 4), (1, 2), False),
            ((2, 3, 4), None, False),
            ((2, 3, 4), None, True),
            ((2, 3, 4, 5), (-2, -1), True),
        ],
    )
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_mean(self, x_shape, axis, keepdim: bool, dtype, compile_fixture):
        np_dtype = np.float32 if dtype == tp.float32 else np.float16
        x = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np_dtype)
        a = tp.Tensor(x, dtype=dtype)
        out = compile_fixture(tp.mean, a, dim=axis, keepdim=keepdim)
        expected = tp.Tensor(cp.array(x.mean(axis=axis, keepdims=keepdim)))
        assert out.shape == expected.shape
        assert tp.allclose(out, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), (1, 2), True),
            ((2, 3, 4), None, False),
            ((2, 3, 4), None, True),
            ((2, 3), 1, False),
            ((2, 3, 4), (1, 2), False),
            ((2, 3, 4, 5), (-2, -1), True),
        ],
    )
    def test_var(self, x_shape, axis, keepdim: bool, compile_fixture):
        x = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32)
        a = tp.Tensor(x)
        out = compile_fixture(tp.var, a, dim=axis, keepdim=keepdim)
        torch_tensor = torch.Tensor(x)
        expected = tp.Tensor(torch_tensor.var(dim=axis, keepdim=keepdim))
        assert out.shape == expected.shape
        assert tp.allclose(out, expected)

    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), 2, True),
            ((2, 3), 1, False),
            ((2, 3, 4), 2, False),
            ((2, 3, 4), None, False),
            ((2, 3, 4), None, True),
        ],
    )
    def test_argmax(self, x_shape, axis, keepdim: bool, compile_fixture):
        x = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32)
        a = tp.Tensor(x)
        out = compile_fixture(tp.argmax, a, dim=axis, keepdim=keepdim)
        assert np.array_equal(cp.from_dlpack(out).get(), np.array(x.argmax(axis=axis, keepdims=keepdim)))

    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), 2, True),
            ((2, 3), 1, False),
            ((2, 3, 4), 2, False),
            ((2, 3, 4), None, False),
            ((2, 3, 4), None, True),
        ],
    )
    def test_argmin(self, x_shape, axis, keepdim: bool, compile_fixture):
        x = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32)
        a = tp.Tensor(x)
        out = compile_fixture(tp.argmin, a, dim=axis, keepdim=keepdim)
        assert np.array_equal(cp.from_dlpack(out).get(), np.array(x.argmin(axis=axis, keepdims=keepdim)))
