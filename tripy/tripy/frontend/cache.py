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
import hashlib
from typing import List

from copy import deepcopy


class ExecutableCache:
    """Global cache for storing compiled executables."""

    def __init__(self):
        self._cache = {}

    def _normalize_trace(self, trace: "Trace") -> str:
        """
        Normalize the trace by renaming all tensor names (inputs, outputs, intermediates)
        and operations to sequential names (t0, t1, ..., tn) while preserving the structure.
        The final key is generated using `str(trace_copy)`.
        """
        tensor_map = {}
        next_tensor_id = 0
        print("before\n", str(trace))

        # Create a shallow copy of the trace
        trace_copy = deepcopy(trace)

        def get_or_assign_tensor_name(tensor):
            """Assign a new name to the tensor or retrieve the existing one."""
            nonlocal next_tensor_id, tensor_map
            t_id = id(tensor)
            if t_id not in tensor_map:
                tensor_map[t_id] = f"t{next_tensor_id}"
                next_tensor_id += 1
            return tensor_map[t_id]

        # Rename inputs
        for inp in trace_copy.inputs:
            inp.name = get_or_assign_tensor_name(inp)

        # Rename operations
        for op in trace_copy.ops:
            for inp in op.inputs:
                inp.name = get_or_assign_tensor_name(inp)

            for out in op.outputs:
                out.name = get_or_assign_tensor_name(out)

        # Rename outputs
        for out in trace_copy.outputs:
            out.name = get_or_assign_tensor_name(out)

        print("after\n", str(trace_copy))
        return str(trace_copy)

    def _generate_key(self, trace: "Trace") -> str:
        """
        Generate a hashable key for the trace by normalizing its entire structure
        and hashing the resulting string.
        """
        normalized_trace = self._normalize_trace(trace)
        return hashlib.sha256(normalized_trace.encode("utf-8")).hexdigest()

    def get(self, trace: "Trace"):
        """Retrieve an executable from the cache."""
        key = self._generate_key(trace)
        return self._cache.get(key, None)

    def set(self, trace: "Trace", value):
        """Store an executable in the cache."""
        key = self._generate_key(trace)
        self._cache[key] = value

    def clear(self):
        """Clear the entire cache."""
        self._cache.clear()

    def size(self) -> int:
        """Return the number of items in the cache."""
        return len(self._cache)

    def get_keys(self):
        """Return all keys currently in the cache."""
        return list(self._cache.keys())
