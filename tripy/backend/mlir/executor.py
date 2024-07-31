#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.runtime.api as runtime

from tripy.backend.utils import TensorInfo
from tripy.common import Array, datatype, device
from tripy.common.exception import raise_error
from tripy.utils import log_time, make_tuple

G_RUNTIME_CLIENT = None


def _get_runtime_client() -> runtime.RuntimeClient:
    global G_RUNTIME_CLIENT
    if G_RUNTIME_CLIENT is None:
        G_RUNTIME_CLIENT = runtime.RuntimeClient()

    return G_RUNTIME_CLIENT


class Executor:
    def __init__(self, executable: runtime.Executable) -> None:
        self.runtime_client = _get_runtime_client()
        self.stream = self.runtime_client.create_stream()
        session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
        self.session = runtime.RuntimeSession(session_options, executable)
        self.device = self.runtime_client.get_devices()[0]  # Assume a single device is available.
        self.signature = executable.get_signature("main")

    def _get_inputs_shape_memref(self, inputs):
        inputs_shape_memref = []
        for input in inputs:
            input_shape = Array(
                input.trace_tensor.producer.data.shape,
                shape=make_tuple(input.trace_tensor.rank),
                dtype=datatype.int64,
                device=device("cpu"),
            )
            inputs_shape_memref.append(input_shape.memref_value)
        return inputs_shape_memref

    def _get_outputs_shape_memref(self):
        offset = self.signature.get_num_input_args()
        outputs_shape_memref = []
        for output_index in range(self.signature.get_num_output_args()):
            arg_index = output_index + offset
            arg = self.signature.get_arg(arg_index)
            assert compiler.MemRefType.isinstance(arg)
            memref = runtime.MemRefType(arg)
            rank = len(memref.shape)
            if len(memref.shape) > 0:
                output_shape = Array(memref.shape, shape=make_tuple(rank), dtype=datatype.int64, device=device("cpu"))
                outputs_shape_memref.append(output_shape.memref_value)
            else:
                outputs_shape_memref.append(None)
        return outputs_shape_memref

    def _execute_shape_inference(self, inputs_shape_memref, outputs_shape_memref):
        # Only execute shape inference if shape function name is valid.
        if not self.signature.get_shape_func_name():
            for memref in outputs_shape_memref:
                if memref is not None:
                    assert (
                        all(dim >= 0 for dim in memref.shape)
                        and f"Output shape {memref.shape} must be statically known if shape inference function is missing. "
                    )
            return None

        self.session.execute_function(
            name=self.signature.get_shape_func_name(), in_args=inputs_shape_memref, out_args=outputs_shape_memref
        )

        outputs_shapes_host = [
            Array(memoryview(s).tolist(), shape=s.shape, dtype=datatype.int64, device=device("cpu"))
            for s in outputs_shape_memref
        ]

        outputs_runtime_shape = [s.data() for s in outputs_shapes_host]
        return outputs_runtime_shape

    def _get_output_tensor_info(self, outputs_runtime_shape, output_devices):
        from tripy.backend.mlir.utils import convert_runtime_dtype_to_tripy_dtype

        offset = self.signature.get_num_input_args()
        outputs_tensor_info = []
        for output_index in range(self.signature.get_num_output_args()):
            arg_index = output_index + offset
            arg = self.signature.get_arg(arg_index)
            assert compiler.MemRefType.isinstance(arg) or compiler.ScalarType.isinstance(
                arg
            ), "Argument must be either MemRefType or ScalarType"
            assert compiler.MemRefType.isinstance(
                arg
            ), "ScalarType argument are not yet supported"  # 158: Add scalar type output argument support.
            memref = compiler.MemRefType(arg)
            dtype = convert_runtime_dtype_to_tripy_dtype(memref.dtype)
            device_type = "gpu" if memref.address_space == runtime.PointerType.device else "cpu"
            if output_devices[output_index]:
                device_type = output_devices[output_index].kind
            is_static_shape = all(dim >= 0 for dim in memref.shape)
            if is_static_shape:
                outputs_tensor_info.append(
                    TensorInfo(len(memref.shape), tuple(memref.shape), dtype, device(device_type))
                )
            else:
                runtime_shape = [
                    rs if dim < 0 else dim for dim, rs in zip(memref.shape, outputs_runtime_shape[output_index])
                ]
                outputs_tensor_info.append(
                    TensorInfo(
                        len(runtime_shape),
                        tuple(runtime_shape),
                        dtype,
                        device(device_type),
                    )
                )
        return outputs_tensor_info

    def get_output_tensor_runtime_info(self, inputs, output_devices=List[device]):
        inputs_shape_memref = self._get_inputs_shape_memref(
            inputs
        )  # Can we use executable signature inputs to retrieve this information?
        outputs_shape_memref = self._get_outputs_shape_memref()
        outputs_runtime_shape = self._execute_shape_inference(inputs_shape_memref, outputs_shape_memref)
        output_tensor_info = self._get_output_tensor_info(outputs_runtime_shape, output_devices)
        return output_tensor_info

    @log_time
    def execute(self, output_devices=List[device], inputs: List["Tensor"] = []) -> List[Array]:
        from tripy.frontend.trace.ops import Storage

        in_args = []
        for inp in inputs:
            assert isinstance(inp.trace_tensor.producer, Storage)
            memref = inp.trace_tensor.producer.data.memref_value
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

        # HACK (#155): Remove `get_devices` once executable output tensor location matches Trace IR.
        out_tensor_info = self.get_output_tensor_runtime_info(inputs, output_devices)

        # Allocate output memory and store buffer pointers.
        outputs = [Array(None, shape=info.shape, dtype=info.dtype, device=info.device) for info in out_tensor_info]

        out_args = []
        for out in outputs:
            memref = out.memref_value
            # HACK (#155): MLIR-TensorRT requires inputs to be on device.
            # Remove explicit copy to device once #155 is addressed.
            if memref.address_space != runtime.PointerType.device:
                memref = self.runtime_client.copy_to_device(
                    host_memref=memref,
                    device=self.runtime_client.get_devices()[0],
                )
            if not memref:
                raise_error("Could not allocate output memref", details=memref.error_details)
            out_args.append(memref)

        # Execute and populate device pointers.
        self.session.execute_function("main", in_args=in_args, out_args=out_args, stream=self.stream)
        self.stream.sync()
        # For outputs that were on the host, do the copy back
        # TODO(#155): MLIR-TensorRT should allow output tensor placements on host.
        for idx, out_info in enumerate(out_tensor_info):
            if out_info.device.kind != "gpu":
                self.runtime_client.copy_to_host(
                    device_memref=out_args[idx],
                    existing_host_memref=outputs[idx].memref_value,
                    stream=None,
                )

        return outputs
