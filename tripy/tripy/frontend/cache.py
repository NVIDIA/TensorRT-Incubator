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
from typing import List, Dict, Any

import mlir_tensorrt.runtime.api as runtime


class ExecutableCache:
    """Global cache for storing compiled executables."""

    def __init__(self):
        self._cache = {}

    def _assign_tensor_name(
        self,
        tensor: "tripy.frontend.trace.tensor.TraceTensor",
        tensor_map: Dict[int, str],
        next_id: List[int],
        backup_map: Dict[int, str] = None,
    ) -> str:
        """
        Assign or retrieve a tensor name.

        Args:
            tensor: The tensor to name
            tensor_map: Mapping of tensor ids to names (clean or original)
            next_id: Mutable list for tracking next tensor ID
            backup_map: Mapping to store original names

        Returns:
            str: The assigned or retrieved tensor name
        """
        t_id = id(tensor)

        # If tensor not in map, assign new name
        if t_id not in tensor_map:
            new_name = f"t{next_id[0]}"
            tensor_map[t_id] = new_name
            if backup_map is not None:
                backup_map[t_id] = tensor.name
            next_id[0] += 1

        return tensor_map[t_id]

    def _update_trace_names(
        self, trace: "Trace", tensor_map: Dict[int, str], next_id: List[int], backup_map: Dict[int, str] = None
    ) -> None:
        """
        Update names for inputs, outputs, and operations in the trace.

        Args:
            trace: The trace to update
            tensor_map: Mapping of tensor ids to names
            next_id: Mutable list for tracking next tensor ID
            backup_map: Mapping of original tensor names
        """
        # Update input names
        for inp in trace.inputs:
            inp.name = self._assign_tensor_name(inp, tensor_map, next_id, backup_map)

        # Update operation input and output names
        for op in trace.ops:
            for inp in op.inputs:
                inp.name = self._assign_tensor_name(inp, tensor_map, next_id, backup_map)
            for out in op.outputs:
                out.name = self._assign_tensor_name(out, tensor_map, next_id, backup_map)

        # Update output names
        for out in trace.outputs:
            out.name = self._assign_tensor_name(out, tensor_map, next_id, backup_map)

    def _normalize_trace(self, trace: "Trace") -> str:
        """
        Normalize the trace by renaming all tensor names while preserving the structure.

        Args:
            trace: The trace to normalize

        Returns:
            str: Normalized trace as a string
        """
        # Initialize maps and next tensor ID
        tensor_map: Dict[int, str] = {}
        backup_tensor_map: Dict[int, str] = {}
        next_tensor_id: List[int] = [0]

        # Clean trace names with sequential names
        self._update_trace_names(trace, tensor_map, next_tensor_id, backup_tensor_map)

        # Get normalized trace string
        trace_str = str(trace)

        # Restore original names
        self._update_trace_names(trace, backup_tensor_map, next_tensor_id)

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
