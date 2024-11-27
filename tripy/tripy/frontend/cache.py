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
from typing import List


class ExecutableCache:
    """Global cache for storing compiled executables."""

    def __init__(self):
        self._cache = {}

    def _normalize_ops(self, ops: List["BaseTraceOp"]) -> str:
        """
        Normalize the key by renaming all tensor names (inputs, outputs, intermediates)
        in the trace to sequential names (t0, t1, ..., tn) while preserving the structure
        of the operations.
        """
        tensor_map = {}
        next_tensor_id = 0

        def get_or_assign_tensor_name(tensor_name: str) -> str:
            """Assign a new name to the tensor or retrieve the existing one."""
            nonlocal next_tensor_id
            if tensor_name not in tensor_map:
                tensor_map[tensor_name] = f"t{next_tensor_id}"
                next_tensor_id += 1
            return tensor_map[tensor_name]

        # Build tensor_map by processing all ops
        for op in ops:
            for t in op.inputs + op.outputs:
                get_or_assign_tensor_name(t.name)

        # Generate normalized string for each op
        normalized_ops = []
        for op in ops:
            op_repr = str(op)
            for old_name, new_name in tensor_map.items():
                op_repr = op_repr.replace(old_name, new_name)  # bug: do shallow copy
            normalized_ops.append(op_repr)

        return "\n".join(normalized_ops)

    def _generate_key(self, trace: "Trace") -> str:
        """
        Generate a hashable key for the trace by normalizing its operations.
        """
        return self._normalize_ops(trace.ops)

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
