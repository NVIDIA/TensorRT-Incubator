#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import nvtripy as tp
from nvtripy.trace.ops.shape import GetDimensionSize, Shape


class TestDimensionSize:
    def test_construction(self):
        s = tp.DimensionSize(1)

        assert isinstance(s, tp.DimensionSize)
        assert s.trace_tensor.producer.inputs == []

    def test_int_conversion(self):
        val = 4
        s = tp.DimensionSize(val)

        assert int(s) == val

    def test_str_method(self):
        s = tp.DimensionSize(12)
        assert str(s) == "12"

    def test_scalar_slice(self):
        a = tp.iota((3, 3))
        assert isinstance(a.shape[0], tp.DimensionSize)

        s = a.shape[0] * a.shape[1]
        b = tp.reshape(a, (s,))
        assert tp.allclose(tp.flatten(a), b)

    def test_scalar_scalar_op(self):
        a = tp.iota((3, 4))
        s1 = a.shape[0]
        s2 = a.shape[1]
        s = s1 + s2
        assert isinstance(s, tp.DimensionSize)

    def test_eval_rebuilds_missing_frontend_tensor_for_shape_input(self):
        a = tp.iota((3, 4))
        s = a.shape[0]

        producer = s.trace_tensor.producer
        assert isinstance(producer, GetDimensionSize)
        assert isinstance(producer.inputs[0].producer, Shape)

        shape_tensor = producer.inputs[0]
        shape_tensor.frontend_tensor = None

        s.eval()

        assert int(s) == 3
