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
import cupy as cp

import nvtripy as tp


def test_default_stream_creation():
    default_stream1 = tp.default_stream()
    default_stream2 = tp.default_stream()
    assert default_stream1 == default_stream2


def test_new_streams():
    stream1 = tp.Stream()
    stream2 = tp.Stream()
    assert stream1 != stream2
    assert stream1.ptr != stream2.ptr
    assert stream1 != tp.default_stream()
    assert stream1.ptr != tp.default_stream().ptr


def test_enqueue_work_on_stream():
    linear = tp.Linear(25, 30)
    linear.weight = tp.ones(linear.weight.shape)
    linear.bias = tp.ones(linear.bias.shape)

    compiled_linear = tp.compile(linear, args=[tp.InputInfo((2, 25), dtype=tp.float32)])

    a = tp.ones((2, 25), dtype=tp.float32).eval()

    out = compiled_linear(a)
    tp.default_stream().synchronize()
    assert tp.equal(out, linear(a))

    stream = tp.Stream()
    compiled_linear.stream = stream
    out = compiled_linear(a)
    # stream sync below is not required since from_dlpack method will eval() the tensor which will call stream sync anyway.
    compiled_linear.stream.synchronize()
    assert tp.equal(out, linear(a))
