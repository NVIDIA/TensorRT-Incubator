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
import pytest

import tripy as tp
import numpy as np


def compare_split_results(tp_out, reference_out):
    if isinstance(tp_out, list):
        assert isinstance(reference_out, tuple)
        assert len(tp_out) == len(reference_out)
        for i in range(len(tp_out)):
            assert cp.array_equal(cp.from_dlpack(tp_out[i]), cp.array(reference_out[i]))
    else:
        assert cp.array_equal(cp.from_dlpack(tp_out), cp.array(reference_out))


class TestSplitOp:
    @pytest.mark.parametrize(
        "dims_a, split_params, reference_slices",
        [
            ((4,), (2, 0), lambda t: (t[:2], t[2:])),
            ((4,), (1, 0), lambda t: t[:]),
            ((4,), (4, 0), lambda t: (t[0:1], t[1:2], t[2:3], t[3:4])),
            ((4,), ([1, 2], 0), lambda t: (t[:1], t[1:2], t[2:])),
            ((12, 12), (3, 1), lambda t: (t[:, :4], t[:, 4:8], t[:, 8:])),
            ((12, 12), ([3], 1), lambda t: (t[:, :3], t[:, 3:])),
            ((12, 12), (4, 0), lambda t: (t[:3, :], t[3:6, :], t[6:9, :], t[9:12, :])),
            ((3, 0), (5, 1), lambda t: (t[:, :0], t[:, 0:0], t[:, 0:0], t[:, 0:0], t[:, 0:0])),
        ],
    )
    def test_split_static(self, dims_a, split_params, reference_slices, compile_fixture):
        a_cp = cp.arange(np.prod(dims_a)).reshape(dims_a).astype(cp.float32)
        a = tp.Tensor(a_cp, device=tp.device("gpu"))

        def func(t):
            return tp.split(t, split_params[0], split_params[1])

        out = compile_fixture(func, a)
        reference_out = reference_slices(a_cp)
        compare_split_results(out, reference_out)
