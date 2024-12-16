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

import pytest

import nvtripy as tp
from nvtripy.frontend.trace.ops import Copy


@pytest.mark.parametrize("src, dst", [("cpu", "gpu"), ("gpu", "cpu")])
def test_copy(src, dst):
    a = tp.Tensor([1, 2], device=tp.device(src))
    a = tp.copy(a, tp.device(dst))
    assert isinstance(a, tp.Tensor)
    assert isinstance(a.trace_tensor.producer, Copy)
    assert a.trace_tensor.producer.target.kind == dst


def test_infer_rank():
    a = tp.Tensor([1, 2], device=tp.device("gpu"))
    a = tp.copy(a, tp.device("cpu"))
    assert a.trace_tensor.rank == 1
