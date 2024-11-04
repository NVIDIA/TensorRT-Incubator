#
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
#

from typing import List

import mlir_tensorrt.runtime.api as runtime
from nvtripy.backend.api.stream import default_stream
from nvtripy.backend.mlir.memref import create_memref
from nvtripy.backend.mlir.utils import MLIRRuntimeClient, convert_runtime_dtype_to_tripy_dtype
from nvtripy.backend.utils import TensorInfo
from nvtripy.common import datatype, device
from nvtripy.common.exception import raise_error
from nvtripy.common.utils import convert_list_to_array
from nvtripy.utils.utils import make_tuple


class Executor:
    def __init__(self, executable: runtime.Executable) -> None:
        self.runtime_client = MLIRRuntimeClient()
        session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
        self.session = runtime.RuntimeSession(session_options, executable)
        self.device = self.runtime_client.get_devices()[0]  # Assume a single device is available.
        self.signature = executable.get_signature("main")
        self.stream = default_stream()

    def execute(self, inputs: List["TraceTensor"] = []) -> List[runtime.MemRefValue]:
        in_args = []
        for inp in inputs:
            memref = inp.producer.data
            # HACK (#155): MLIR-TensorRT requires inputs to be on device.
            # Remove explicit copy to device once #155 is addressed.
            if memref.address_space != runtime.PointerType.device:
                memref = self.runtime_client.copy_to_device(
                    host_memref=memref,
                    device=self.runtime_client.get_devices()[0],
                )
            if not memref:
                raise_error(
                    "Could not convert tensor to memref",
                    details=[f"Tensor was: ", inp, "Error was: ", memref.error_details],
                )
            in_args.append(memref)

        # Execute and populate device pointers.
        outputs = self.session.execute_function(
            "main", in_args=in_args, stream=self.stream._active_cuda_stream, client=self.runtime_client
        )

        return outputs
