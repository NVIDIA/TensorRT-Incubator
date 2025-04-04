#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math

import numpy as np
import nvtripy as tp
import pytest


class TestReduceOp:
    @pytest.mark.parametrize(
        "func, np_func_name",
        [
            (tp.sum, "sum"),
            (tp.prod, "prod"),
            (tp.mean, "mean"),
            (tp.max, "max"),
            (tp.min, "min"),
            (tp.var, "var"),
            (tp.argmax, "argmax"),
            (tp.argmin, "argmin"),
            (tp.any, "any"),
            (tp.all, "all"),
        ],
    )
    @pytest.mark.parametrize("keepdim", [True, False])
    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            # Scalar:
            (tuple(), None),
            # Single dimension:
            ((2, 3), 1),
            # Multiple dimensions:
            ((2, 3, 4), (1, -1)),
            # Default dimensions:
            ((2, 3), None),
        ],
    )
    def test_reduce_ops(self, func, np_func_name, input_shape, dim, keepdim, eager_or_compiled):
        if func in (tp.argmax, tp.argmin) and isinstance(dim, tuple):
            pytest.skip("argmax/argmin do not support multiple dimensions")

        if func in (tp.any, tp.all):
            input = np.array([i % 2 == 0 for i in range(math.prod(input_shape))]).reshape(input_shape)
        else:
            input = np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.float32)

        out = eager_or_compiled(func, tp.Tensor(input), dim=dim, keepdim=keepdim)

        kwargs = {}
        if func is tp.var and math.prod(input_shape) > 1:
            # We need to set the correction factor to match Tripy, except in the scalar case.
            kwargs["ddof"] = 1
        expected = tp.Tensor(np.array(getattr(input, np_func_name)(axis=dim, keepdims=keepdim, **kwargs)))

        assert out.shape == expected.shape

        if not issubclass(out.dtype, tp.floating):
            # Upcast the data type since NumPy sometimes uses int64 instead of int32:
            assert out.dtype == expected.dtype or (expected.dtype == tp.int64 and out.dtype == tp.int32)
            assert tp.equal(tp.cast(out, expected.dtype), expected)
        else:
            assert tp.allclose(out, expected, rtol=1e-3, atol=1e-3)
