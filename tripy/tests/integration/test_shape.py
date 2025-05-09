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

import nvtripy as tp


class TestShape:
    def test_shape(self):
        a = tp.Tensor([[1, 2], [3, 4]])
        shape_a = a.shape
        assert isinstance(a, tp.Tensor)
        assert isinstance(shape_a, tuple)

        assert shape_a == (2, 2)

    def test_static_shape_is_not_mutable(self):
        # a.shape is not mutable, so there's no risk of accidentally
        # modifying the underlying trace tensor shape for static shape tensors.
        a = tp.Tensor(([1, 2]))

        assert a.trace_tensor.shape == (2,)
        assert a.shape == (2,)

        assert isinstance(a.shape, tuple)
