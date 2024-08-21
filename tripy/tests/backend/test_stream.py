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
import pytest
import gc, sys
import tripy as tp
import cupy as cp


def test_default_stream_creation():
    default_stream1 = tp.Stream(create_new=False)
    default_stream2 = tp.Stream(create_new=False)
    assert default_stream1 == default_stream2


def test_get_all_streams():
    tp.Stream._all_streams.clear()
    stream1 = tp.Stream()
    stream2 = tp.Stream()
    all_streams = tp.Stream.get_all_streams()
    assert len(all_streams) == 2
    assert stream1 in all_streams
    assert stream2 in all_streams


def test_enqueue_work_on_stream():
    linear = tp.Linear(25, 30)
    compiler = tp.Compiler(linear)

    compiled_linear = compiler.compile(tp.InputInfo((2, 25), dtype=tp.float32))

    a = tp.ones((2, 25), dtype=tp.float32)

    stream = tp.Stream()
    compiled_linear.stream = stream
    out = compiled_linear(a)
    # stream sync below is not required since from_dlpack method will eval() the tensor which will call stream sync anyway.
    compiled_linear.stream.synchronize()
    assert cp.array_equal(cp.from_dlpack(out), cp.from_dlpack(linear(a)))
