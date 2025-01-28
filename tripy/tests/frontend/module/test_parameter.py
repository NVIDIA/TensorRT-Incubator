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

import nvtripy as tp
from nvtripy.frontend.module.parameter import DefaultParameter
from nvtripy.trace.ops.storage import Storage


class TestDefaultParameter:
    def test_shape_dtype_access_does_not_materialize_data(self):
        param = DefaultParameter((1, 2), dtype=tp.float32)
        assert param.shape == [1, 2]
        assert type(param.shape[0]) is int  # Make sure we are not compiling and getting DimensionSizes

        assert param.dtype == tp.float32
        assert not isinstance(param.trace_tensor.producer, Storage)

    def test_data_can_be_materialized(self):
        param = DefaultParameter((1, 2), dtype=tp.float32)
        assert np.array_equal(cp.from_dlpack(param).get(), np.array([[0, 1]], dtype=np.float32))
