#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain input copy of the License at
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
import torch


class TestSliceOp:
    @pytest.mark.parametrize("use_constant", [True, False])
    @pytest.mark.parametrize(
        "shape, slice_func",
        [
            ((2,), lambda t: t[-1]),
            ((4,), lambda t: t[-2]),
            ((4,), lambda t: t[1:]),
            ((2, 3, 4), lambda t: t[1, 2, 3]),
            ((2, 3, 4), lambda t: t[:, ::-1, :]),
            ((2, 3, 4), lambda t: t[:, :, ::-2]),
            ((10,), lambda t: t[8:2:-2]),
            ((10,), lambda t: t[:2:-2]),
            ((10,), lambda t: t[10:0:-1]),
            ((10,), lambda t: t[1024:0:-1]),
            ((1, 2, 1, 4), lambda t: t[:, 1, 0, 2:-1]),
            ((2, 3, 4), lambda t: t[:3, :4, :5]),
            ((2, 3, 4), lambda t: t[0:0, :-2, 1:]),
            ((10,), lambda t: t[1234:5678]),
            ((10,), lambda t: t[1234:5678:-1]),
            ((5,), lambda t: t[-5:-12]),
            ((5,), lambda t: t[-12:-5:-1]),
            # Empty slices
            ((10,), lambda t: t[5:2:1]),  # Positive step size with start > stop
            ((5, 5), lambda t: t[4:1:2, :]),
            ((10,), lambda t: t[2:5:-1]),  # Negative step size with stop > start
            ((5, 5), lambda t: t[1:4:-1, :]),
            ((3, 4, 5), lambda t: t[2:1, 3:2, 4:3]),  # Multi-dimensional empty slices
            ((3, 4, 5), lambda t: t[1:2, 3:2, ::-1]),
            # Omitted start, stop, and step
            ((10,), lambda t: t[:]),  # Omit all
            ((10,), lambda t: t[2:]),  # Omit stop and step
            ((10,), lambda t: t[:8]),  # Omit start and step
            ((10,), lambda t: t[::2]),  # Omit start and stop
            ((10,), lambda t: t[2:8]),  # Omit step
            ((10,), lambda t: t[2::2]),  # Omit stop
            ((10,), lambda t: t[:8:2]),  # Omit start
        ],
    )
    def test_slice(self, use_constant, shape, slice_func, eager_or_compiled):
        if use_constant:
            input_np = np.arange(math.prod(shape)).reshape(shape).astype(np.float32)
            input = tp.Tensor(input_np, device=tp.device("gpu"))
        else:
            input = tp.reshape(tp.arange(math.prod(shape)), shape)
            input_np = np.from_dlpack(tp.copy(input, device=tp.device("cpu")))

        out = eager_or_compiled(slice_func, input)
        assert out.shape == slice_func(input_np).shape
        assert np.array_equal(np.from_dlpack(tp.copy(out, device=tp.device("cpu"))), slice_func(input_np).get())

    @pytest.mark.parametrize("use_constant", [True, False])
    @pytest.mark.parametrize(
        "shape, slice_func, reference_func",
        [
            ((10,), lambda t: t[tp.DimensionSize(2) : tp.DimensionSize(8) : tp.DimensionSize(2)], lambda t: t[2:8:2]),
            ((10,), lambda t: t[tp.DimensionSize(8) : tp.DimensionSize(2) : tp.DimensionSize(-2)], lambda t: t[8:2:-2]),
            ((10,), lambda t: t[tp.DimensionSize(2) :: tp.DimensionSize(2)], lambda t: t[2::2]),
            ((10,), lambda t: t[: tp.DimensionSize(8) : tp.DimensionSize(2)], lambda t: t[:8:2]),
            ((10,), lambda t: t[:], lambda t: t[:]),  # Omit all
            ((10,), lambda t: t[tp.DimensionSize(2) :], lambda t: t[2:]),  # Omit stop and step
            ((10,), lambda t: t[: tp.DimensionSize(8)], lambda t: t[:8]),  # Omit start and step
            ((10,), lambda t: t[:: tp.DimensionSize(2)], lambda t: t[::2]),  # Omit start and stop
            ((10,), lambda t: t[tp.DimensionSize(2) : tp.DimensionSize(8)], lambda t: t[2:8]),  # Omit step
            ((10,), lambda t: t[tp.DimensionSize(2) :: tp.DimensionSize(2)], lambda t: t[2::2]),  # Omit stop
            ((10,), lambda t: t[: tp.DimensionSize(8) : tp.DimensionSize(2)], lambda t: t[:8:2]),  # Omit start
        ],
    )
    def test_slice_with_dimensionsize(self, use_constant, shape, slice_func, reference_func, eager_or_compiled):
        if use_constant:
            input_np = np.arange(math.prod(shape)).reshape(shape).astype(np.float32)
            input = tp.Tensor(input_np, device=tp.device("gpu"))
        else:
            input = tp.reshape(tp.arange(math.prod(shape)), shape)
            input_np = np.from_dlpack(tp.copy(input, device=tp.device("cpu")))

        out = eager_or_compiled(slice_func, input)
        assert out.shape == reference_func(input_np).shape
        assert np.array_equal(np.from_dlpack(tp.copy(out, device=tp.device("cpu"))), reference_func(input_np).get())

    @pytest.mark.parametrize(
        "input_shape, index_shape",
        [
            ((3,), (3,)),  # Simple gather
            ((4, 3, 2), (2, 2)),  # Multi-dimensional gather
        ],
    )
    def test_slice_as_gather(self, input_shape, index_shape, eager_or_compiled):
        input = tp.reshape(tp.arange(math.prod(input_shape)), input_shape)
        index = tp.reshape(tp.arange(math.prod(index_shape), dtype=tp.int32), index_shape)

        def slice(input, index):
            return input[index]

        output = eager_or_compiled(slice, input, index)
        assert torch.equal(torch.from_dlpack(output), torch.from_dlpack(input)[torch.from_dlpack(index)])
