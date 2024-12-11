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

import mlir_tensorrt.runtime.api as runtime


class ExecutableCache:
    """Global cache for storing compiled executables."""

    def __init__(self):
        self._cache = {}

    def _normalize_trace(self, trace: "Trace") -> str:
        """
        Normalize the trace by renaming all tensor names (inputs, outputs, intermediates)
        and operations to sequential names (t0, t1, ..., tn) while preserving the structure.
        Use a clean function to avoid deep copies and return the normalized trace as a string.
        """
        clean_tensor_map = {}
        original_tensor_map = {}
        next_tensor_id = 0

        original_str = str(trace)

        def get_or_assign_tensor_name(tensor):
            """Assign a new name to the tensor or retrieve the existing one."""
            nonlocal next_tensor_id, clean_tensor_map, original_tensor_map
            t_id = id(tensor)
            if t_id not in clean_tensor_map:
                clean_name = f"t{next_tensor_id}"
                clean_tensor_map[t_id] = clean_name
                original_tensor_map[t_id] = tensor.name
                next_tensor_id += 1
            return clean_tensor_map[t_id]

        def clean_trace(trace, tensor_map):
            """Rename tensors in the trace using the provided map."""
            for inp in trace.inputs:
                inp.name = get_or_assign_tensor_name(inp)
            for op in trace.ops:
                for inp in op.inputs:
                    inp.name = get_or_assign_tensor_name(inp)
                for out in op.outputs:
                    out.name = get_or_assign_tensor_name(out)
            for out in trace.outputs:
                out.name = get_or_assign_tensor_name(out)

        def restore_original_names(trace, tensor_map):
            """Restore the original tensor names using the map."""
            for inp in trace.inputs:
                inp.name = tensor_map.get(id(inp), inp.name)
            for op in trace.ops:
                for inp in op.inputs:
                    inp.name = tensor_map.get(id(inp), inp.name)
                for out in op.outputs:
                    out.name = tensor_map.get(id(out), out.name)
            for out in trace.outputs:
                out.name = tensor_map.get(id(out), out.name)

        clean_trace(trace, clean_tensor_map)
        trace_str = str(trace)
        restore_original_names(trace, original_tensor_map)

        return trace_str

    def _generate_key(self, trace: "Trace", devices: List["tripy.common.device"]) -> str:
        normalized_trace = self._normalize_trace(trace)
        key = normalized_trace + "\ndevices:\n" + "\n".join([str(device) for device in devices])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get(self, trace: "Trace", devices: List["tripy.common.device"]):
        key = self._generate_key(trace, devices)
        return self._cache.get(key, None)

    def set(self, trace: "Trace", executable: runtime.Executable, devices: List["tripy.common.device"]):
        key = self._generate_key(trace, devices)
        self._cache[key] = executable

    def size(self) -> int:
        """Return the number of items in the cache."""
        return len(self._cache)
