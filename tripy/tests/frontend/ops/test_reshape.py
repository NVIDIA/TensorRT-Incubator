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


import nvtripy as tp
from tests import helper


class TestReshape:
    def test_invalid_neg_dim_reshape(self):
        shape = (1, 30)
        new_shape = (-1, -1)
        with helper.raises(tp.TripyException, match="The new shape can have at most one inferred dimension"):
            a = tp.reshape(tp.ones(shape), new_shape)
            print(a)
