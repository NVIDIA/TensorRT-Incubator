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

import cupy as cp
import numpy as np
import nvtripy as tp
import pytest
from nvtripy.common.datatype import DATA_TYPES
from nvtripy.frontend.ops import utils as op_utils
from tests import helper


class TestIota:
    DTYPE_PARAMS = [
        (("float32", tp.common.datatype.float32)),
        (("float16", tp.common.datatype.float16)),
        (("int32", tp.common.datatype.int32)),
    ]

    def _compute_ref_iota(self, dtype, shape, dim):
        if dim is None:
            dim = 0
        elif dim < 0:
            dim += len(shape)
        expected = np.arange(0, shape[dim], dtype=dtype)
        if dim < len(shape) - 1:
            expand_dims = [1 + i for i in range(len(shape) - 1 - dim)]
            expected = np.expand_dims(expected, expand_dims)
        expected = np.broadcast_to(expected, shape)
        return expected

    @pytest.mark.parametrize("dtype", DTYPE_PARAMS)
    @pytest.mark.parametrize(
        "shape, dim",
        [
            ((2, 3), 1),
            ((2, 3), 0),
            ((2, 3), -1),
            ((2, 3, 4), 2),
        ],
    )
    def test_iota(self, dtype, shape, dim, eager_or_compiled):
        output = eager_or_compiled(tp.iota, shape, dim, dtype[1])
        assert np.array_equal(
            np.from_dlpack(tp.copy(output, device=tp.device("cpu"))), self._compute_ref_iota(dtype[0], shape, dim)
        )

    @pytest.mark.parametrize("dtype", DTYPE_PARAMS)
    @pytest.mark.parametrize(
        "shape, dim",
        [
            ((2, 3), 1),
            ((2, 3), None),
            ((2, 3), -1),
            ((2, 3, 4), 2),
        ],
    )
    def test_iota_like(self, dtype, shape, dim, eager_or_compiled):
        if dim:
            output = eager_or_compiled(tp.iota_like, tp.ones(shape), dim, dtype[1])
        else:
            output = eager_or_compiled(tp.iota_like, tp.ones(shape), dtype=dtype[1])

        assert np.array_equal(
            np.from_dlpack(tp.copy(output, device=tp.device("cpu"))), self._compute_ref_iota(dtype[0], shape, dim)
        )

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_negative_no_casting(self, dtype):
        from nvtripy.trace.ops.linspace import Linspace

        if Linspace.get_closest_dtype(dtype) == dtype:
            pytest.skip(f"tp.iota() supports {dtype} without cast")

        # TODO: update the 'match' error msg when MLIR-TRT fixes dtype constraint
        a = tp.ones((2, 2))
        shape = tp.Tensor([2, 2])
        step = tp.Tensor([2, 1])
        out = op_utils.create_op(Linspace, [op_utils.tensor_from_shape_like(a.shape), shape, step], dtype=dtype)

        exception_str = "failed to run pass pipeline"
        with helper.raises(
            tp.TripyException,
            match=exception_str,
        ):
            print(out)

    def test_iota_from_shape_tensor(self, eager_or_compiled):
        a = tp.ones((2, 2))
        output = eager_or_compiled(tp.iota, a.shape)
        assert np.array_equal(
            np.from_dlpack(tp.copy(output, device=tp.device("cpu"))), self._compute_ref_iota("float32", (2, 2), 0)
        )

    def test_iota_from_mixed_seqence(self, eager_or_compiled):
        a = tp.ones((2, 2))
        output = eager_or_compiled(tp.iota, (3, a.shape[0]))
        assert np.array_equal(
            np.from_dlpack(tp.copy(output, device=tp.device("cpu"))), self._compute_ref_iota("float32", (3, 2), 0)
        )
