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
from typing import List, Dict, Optional


from nvtripy import utils
from nvtripy import config


class ExecutableCache:
    """Global cache for storing compiled executables."""

    def __init__(self):
        self._cache: Dict[str, "nvtripy.Executable"] = {}

    def _assign_tensor_name(
        self,
        tensor: "nvtripy.trace.tensor.TraceTensor",
        tensor_map: Dict[int, str],
        backup_map: Dict[int, str] = None,
    ) -> str:
        """
        Assign or retrieve a tensor name.

        Args:
            tensor (TraceTensor): The tensor to name.
            tensor_map (Dict[int, str]): Mapping of tensor ids to names (clean or original).
            backup_map (Dict[int, str], optional): Mapping to store original names. Defaults to None.

        Returns:
            str: The assigned or retrieved tensor name.
        """
        t_id = id(tensor)

        # If tensor not in map, assign new name
        if t_id not in tensor_map:
            new_name = f"t{len(tensor_map)}"
            tensor_map[t_id] = new_name
            if backup_map is not None:
                backup_map[t_id] = tensor.name

        return tensor_map[t_id]

    def _update_trace_names(
        self, trace: "Trace", tensor_map: Dict[int, str], backup_map: Dict[int, str] = None
    ) -> None:
        """
        Update names for inputs, outputs, and operations in the trace.

        Args:
            trace (Trace): The trace to update.
            tensor_map (Dict[int, str]): Mapping of tensor ids to names.
            backup_map (Dict[int, str], optional): Mapping of original tensor names. Defaults to None.
        """
        # Update input names
        for inp in trace.inputs:
            inp.name = self._assign_tensor_name(inp, tensor_map, backup_map)

        # Update operation input and output names
        for op in trace.ops:
            for inp in op.inputs:
                inp.name = self._assign_tensor_name(inp, tensor_map, backup_map)
            for out in op.outputs:
                out.name = self._assign_tensor_name(out, tensor_map, backup_map)

        # Update output names
        for out in trace.outputs:
            out.name = self._assign_tensor_name(out, tensor_map, backup_map)

    def _normalize_trace(self, trace: "Trace") -> str:
        """
        Normalize the trace by renaming all tensor names while preserving the structure.

        Args:
            trace (Trace): The trace to normalize.

        Returns:
            str: Normalized trace as a string.
        """
        # Initialize maps
        tensor_map: Dict[int, str] = {}
        backup_tensor_map: Dict[int, str] = {}

        # Clean trace names with sequential names
        self._update_trace_names(trace, tensor_map, backup_tensor_map)

        # Get normalized trace string
        trace_str = str(trace)  # TODO (#467): Add custom context manager

        # Restore original names
        self._update_trace_names(trace, backup_tensor_map)

        return trace_str

    def _generate_key(self, trace: "Trace", devices: List["nvtripy.device"]) -> str:
        """
        Generate a unique key for a given trace and device configuration.

        Args:
            trace (Trace): The trace for which to generate the key.
            devices (List[Device]): List of devices associated with the trace.

        Returns:
            str: A unique hash key representing the trace and devices.
        """
        normalized_trace = self._normalize_trace(trace)
        key = normalized_trace + "\ndevices:\n" + "\n".join([str(device) for device in devices])
        return utils.utils.md5(key.encode("utf-8"))

    # TODO (pranavm): This is only here because of the tests - rewrite this and update the tests
    def get(self, trace, devices):
        key = self._generate_key(trace, devices)
        return self._cache.get(key)

    # TODO (pranavm): Update integration tests for new behavior of this method.
    def compile(self, trace: "Trace", devices: List["nvtripy.device"]) -> Optional["nvtripy.Executable"]:
        """
        Retrieve a cached executable for the given trace and devices or compile a new one.

        Args:
            trace (Trace): The trace used as a key.
            devices (List[Device]): List of devices associated with the trace.

        Returns:
            Executable: The executable.
        """

        key = self._generate_key(trace, devices)

        if key not in self._cache:
            from nvtripy.backend.api.executable import Executable
            from nvtripy.backend.mlir.compiler import Compiler

            mlir = trace.to_mlir()

            compiler = Compiler(trt_builder_opt_level=0)
            # TODO (pranavm): Add error mapping logic here (test with squeezing non-singleton dim)
            arg_names = [f"arg{index}" for index in range(len(trace.inputs))]
            executable = Executable(compiler.compile(mlir, trace=trace), arg_names)

            if not config.use_cache_in_eager_mode:
                return executable

            self._cache[key] = executable

        return self._cache[key]


global_cache = ExecutableCache()
