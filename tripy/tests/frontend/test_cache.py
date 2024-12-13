# # SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# import pytest

# import tripy as tp

# from tripy.frontend.trace import Trace
# from tripy.frontend.cache import ExecutableCache


# @pytest.fixture
# def cache():
#     return ExecutableCache()


# class TestCache:
#     def test_get_nonexistent_key(self, cache):
#         """Test getting a value for a nonexistent key."""
#         cached_value = cache.get("nonexistent_key")

#         assert cached_value is None, "Expected None for a nonexistent key"

#     def test_normalize_key(self, cache):
#         raw_trace =
#         expected_key =
#         assert cache._normalize_key(raw_key) == expected_key

#     def test_same_operation_different_tensor(self, cache, monkeypatch):
#         # Mock the global cache
#         monkeypatch.setattr(tp.frontend, "global_cache", cache)

#         tensor = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float32)

#         cache.clear()  # Ensure the cache is empty
#         assert cache.size() == 0

#         tensor.eval()

#         tensor2 = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float32)
#         assert cache.size() == 1
#         assert cache.get(str(Trace([tensor2]))) is not None  # Ensure tensor trace is in cache

#     def test_equivalance(self, cache, monkeypatch):
#         # Mock the global cache
#         monkeypatch.setattr(tp.frontend, "global_cache", cache)

#         input_tensor = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float32)
#         layer = tp.Linear(2, 3)

#         # Operation without caching
#         output_without_cache = layer(input_tensor)

#         cache.clear()  # Ensure the cache is empty
#         assert cache.size() == 0

#         # Eval operation without caching
#         output_without_cache.eval()

#         # Operation with caching
#         output_with_cache = layer(input_tensor)
#         assert cache.get(str(Trace([output_with_cache]))) is not None

#         # Run the operation with caching
#         output_with_cache.eval()

#         # Assert outputs are equivalent
#         assert tp.allclose(output_without_cache, output_with_cache)
