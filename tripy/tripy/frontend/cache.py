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
import re


class ExecutableCache:
    """Global cache for storing compiled executables."""

    def __init__(self):
        self._cache = {}

    def _normalize_key(self, key: str) -> str:
        """Normalize the key by ordering all tensors with names matching t\d*
        and renaming them sequentially as t0, t1, t2, etc.
        """
        # Find all tensor identifiers (e.g., t90, t91)
        tensors = sorted(set(re.findall(r"t\d+", key)), key=lambda x: int(x[1:]))

        # Map old tensor names to new sequential names
        tensor_map = {tensor: f"t{i}" for i, tensor in enumerate(tensors)}

        # Replace old tensor names in the key with the normalized names
        normalized_key = key
        for old, new in tensor_map.items():
            normalized_key = re.sub(rf"\b{old}\b", new, normalized_key)

        return normalized_key

    def get(self, key: str):
        """Retrieve an executable from the cache."""
        normalized_key = self._normalize_key(key)
        return self._cache.get(normalized_key, None)

    def set(self, key: str, value):
        """Store an executable in the cache."""
        normalized_key = self._normalize_key(key)
        self._cache[normalized_key] = value

    def clear(self):
        """Clear the entire cache."""
        self._cache.clear()

    def size(self) -> int:
        """Return the number of items in the cache."""
        return len(self._cache)

    def get_keys(self):
        """Return all keys currently in the cache."""
        return list(self._cache.keys())
