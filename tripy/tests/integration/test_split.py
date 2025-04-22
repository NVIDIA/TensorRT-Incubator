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


def compare_split_results(tp_outs, ref_outs):
    assert isinstance(ref_outs, tuple)
    assert len(tp_outs) == len(ref_outs)
    for tp_out, ref_out in zip(tp_outs, ref_outs):
        assert cp.array_equal(cp.from_dlpack(tp_out), cp.array(ref_out))


class TestSplitOp:
    @pytest.mark.parametrize(
        "dims_a, num_split_or_sizes, dim, reference_slices",
        [
            ((4,), 2, 0, lambda t: (t[:2], t[2:])),
            ((4,), 1, 0, lambda t: (t[:],)),
            ((4,), 4, 0, lambda t: (t[0:1], t[1:2], t[2:3], t[3:4])),
            ((4,), [1, 1, 2], 0, lambda t: (t[:1], t[1:2], t[2:])),
            ((5,), 2, 0, lambda t: (t[:3], t[3:])),
            ((12, 12), 3, 1, lambda t: (t[:, :4], t[:, 4:8], t[:, 8:])),
            ((12, 12), [3, 9], 1, lambda t: (t[:, :3], t[:, 3:])),
            ((12, 12), [tp.DimensionSize(3), 9], 1, lambda t: (t[:, :3], t[:, 3:])),
            ((12, 12), 4, 0, lambda t: (t[:3, :], t[3:6, :], t[6:9, :], t[9:12, :])),
            pytest.param(
                (3, 0),
                5,
                1,
                lambda t: (t[:, :0], t[:, 0:0], t[:, 0:0], t[:, 0:0], t[:, 0:0]),
                marks=pytest.mark.skip(reason="https://github.com/NVIDIA/TensorRT-Incubator/issues/398"),
            ),
        ],
    )
    def test_split(self, dims_a, num_split_or_sizes, dim, reference_slices, eager_or_compiled):
        a_cp = cp.arange(np.prod(dims_a)).reshape(dims_a).astype(cp.float32)
        a = tp.Tensor(a_cp, device=tp.device("gpu"))

        def func(t):
            return tp.split(t, num_split_or_sizes, dim)

        outs = eager_or_compiled(func, a)
        ref_outs = reference_slices(a_cp)
        compare_split_results(outs, ref_outs)
