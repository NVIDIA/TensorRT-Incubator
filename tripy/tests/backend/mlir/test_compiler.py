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

import pytest
from tests import helper

import nvtripy as tp


class TestCompilerClient:
    def test_concurrent_compiler_init(self):
        import concurrent.futures
        from nvtripy.backend.mlir.compiler import _get_compiler_objects

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_get_compiler_objects) for _ in range(4)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        # All threads should get back the same compiler client instance.
        clients = [client for _, client in results]
        assert all(c is clients[0] for c in clients), "All threads should share the same CompilerClient"


# Tests to ensure that we're able to map errors from MLIR-TRT back to the Python code cleanly.
class TestErrorMapping:
    def test_invalid_slice(self):
        values = tp.Tensor([1, 2, 3])
        sliced = values[4]

        with helper.raises(tp.TripyException, r"out of bounds access"):
            sliced.eval()

    def test_reshape_invalid_volume(self):
        tensor = tp.ones((2, 2))
        reshaped = tp.reshape(tensor, (3, 3))

        with helper.raises(tp.TripyException, r"reshape changes volume", has_stack_info_for=[reshaped]):
            reshaped.eval()
