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

import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.runtime.api as runtime

from tripy.backend.mlir.memref import create_memref
from tripy.backend.utils import TensorInfo
from tripy.common import datatype, device
from tripy.common.exception import raise_error
from tripy.utils import log_time, make_tuple


class Executor:
    def __init__(self, executable: runtime.Executable) -> None:
        from tripy.backend.api.stream import default_stream
        from tripy.backend.mlir.utils import MLIRRuntimeClient

        self.runtime_client = MLIRRuntimeClient()
        session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
        self.session = runtime.RuntimeSession(session_options, executable)
        self.device = self.runtime_client.get_devices()[0]  # Assume a single device is available.
        self.signature = executable.get_signature("main")
        self.stream = default_stream()

    def _create_shape_memref(self, shape):
        from tripy.common.utils import convert_list_to_array

        shape = make_tuple(shape)
        if len(shape) == 0:
            return create_memref(
                shape=(0,),
                dtype=datatype.int64,
                device=device("cpu"),
            )
        return create_memref(
            array=convert_list_to_array(shape, datatype.int64),
            shape=(len(shape),),
            dtype=datatype.int64,
            device=device("cpu"),
        )

    def _get_inputs_runtime_shape(self, inputs):
        inputs_shape = []
        for input in inputs:
            inputs_shape.append(input.trace_tensor.producer.data.shape)
        return inputs_shape

    def _get_outputs_shape(self):
        offset = self.signature.get_num_input_args()
        outputs_shape = []
        all_outputs_known = True
        for output_index in range(self.signature.get_num_output_args()):
            arg_index = output_index + offset
            arg = self.signature.get_arg(arg_index)
            assert compiler.MemRefType.isinstance(arg)
            memref = runtime.MemRefType(arg)
            rank = len(memref.shape)

            outputs_shape.append(memref.shape)
            if rank > 0:
                all_outputs_known &= all(dim >= 0 for dim in memref.shape)
        return outputs_shape, all_outputs_known

    def _execute_shape_inference(self, inputs_shape, outputs_shape):
        # Only execute shape inference if shape function name is valid.
        assert (
            self.signature.get_shape_func_name()
        ), f"Shape inference function is missing while output shapes are not known."

        inputs_shape_memref = [self._create_shape_memref(inp_shape) for inp_shape in inputs_shape]
        outputs_shape_memref = [self._create_shape_memref(out_shape) for out_shape in outputs_shape]
        self.session.execute_function(
            name=self.signature.get_shape_func_name(), in_args=inputs_shape_memref, out_args=outputs_shape_memref
        )

        outputs_runtime_shape = [memoryview(s).tolist() for s in outputs_shape_memref]
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
        outputs_shape, all_outputs_known = self._get_outputs_shape()
        if not all_outputs_known:
            inputs_shape = self._get_inputs_runtime_shape(inputs)
            outputs_shape = self._execute_shape_inference(inputs_shape, outputs_shape)
        output_tensor_info = self._get_output_tensor_info(outputs_shape, output_devices)
        return output_tensor_info

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, stream):
        self._stream = stream

    @log_time
    def execute(self, output_devices=List[device], inputs: List["Tensor"] = []) -> List[runtime.MemRefValue]:
        from tripy.frontend.trace.ops import Storage

        in_args = []
        for inp in inputs:
            assert isinstance(inp.trace_tensor.producer, Storage) and inp.trace_tensor.producer.has_memref
            memref = inp.trace_tensor.producer.data
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
        outputs = [
            create_memref(
                shape=info.shape, dtype=info.dtype, device=info.device, stream=self.stream._active_cuda_stream
            )
            for info in out_tensor_info
        ]

        out_args = []
        for out in outputs:
            memref = out
            # HACK (#155): MLIR-TensorRT requires inputs to be on device.
            # Remove explicit copy to device once #155 is addressed.
            if memref.address_space != runtime.PointerType.device:
                memref = self.runtime_client.copy_to_device(
                    host_memref=memref,
                    device=self.runtime_client.get_devices()[0],
                    stream=self.stream._active_cuda_stream,
                )
            if not memref:
                raise_error("Could not allocate output memref", details=memref.error_details)
            out_args.append(memref)

        # Execute and populate device pointers.
        self.session.execute_function(
            "main", in_args=in_args, out_args=out_args, stream=self.stream._active_cuda_stream
        )

        # For outputs that were on the host, do the copy back
        # TODO(#155): MLIR-TensorRT should allow output tensor placements on host.
        for idx, out_info in enumerate(out_tensor_info):
            if out_info.device.kind != "gpu":
                self.runtime_client.copy_to_host(
                    device_memref=out_args[idx],
                    existing_host_memref=outputs[idx],
                    stream=self.stream._active_cuda_stream,
                )

        return outputs
