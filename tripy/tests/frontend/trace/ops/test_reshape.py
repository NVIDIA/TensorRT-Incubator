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


from tests import helper

import tripy as tp
from tripy.frontend.trace.ops import Reshape


class TestReshape:
    def test_op_func(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = tp.reshape(a, (1, 1, 4))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Reshape)

    def test_neg_dim_func(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = tp.reshape(a, (1, 1, -1))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Reshape)

    def test_invalid_neg_dim_reshape(self):
        shape = (1, 30)
        new_shape = (-1, -1)
        with helper.raises(tp.TripyException, match="The new shape can have at most one inferred dimension"):
            a = tp.reshape(tp.ones(shape), new_shape)
            print(a)

    def test_infer_rank(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = tp.reshape(a, (1, 1, -1))
        assert a.trace_tensor.rank == 3
