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

from dataclasses import dataclass
from typing import List, Sequence, Set, Any

import mlir_tensorrt.runtime.api as runtime

from nvtripy import utils
from nvtripy.backend.mlir import memref
from nvtripy.backend.mlir import utils as mlir_utils
from nvtripy.common import datatype
from nvtripy.common import device as tp_device
from nvtripy.common import utils as common_utils
from nvtripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Storage(BaseTraceOp):

    data: runtime.MemRefValue
    shape: Sequence[int]
    dtype: type
    device: tp_device
    data_str: str = ""

    def __init__(
        self,
        inputs: List["Tensor"],
        outputs: List["Tensor"],
        data: Any,
        device: tp_device = None,
    ) -> None:
        super().__init__(inputs, outputs)

        original_data = data

        # Handle if data is dlpacked but not memref yet
        if hasattr(data, "__dlpack__") and not isinstance(data, runtime.MemRefValue):
            data = memref.create_memref_view(data)

        if isinstance(data, runtime.MemRefValue):
            self.data = data
            self.dtype = mlir_utils.convert_runtime_dtype_to_tripy_dtype(self.data.dtype)
            self.shape = tuple(data.shape)
            self.device = tp_device.create_directly(
                "gpu" if data.address_space == runtime.PointerType.device else "cpu", 0
            )
        else:
            if common_utils.is_empty(data):
                self.dtype = datatype.float32
                data_array = None
            else:
                self.dtype = common_utils.get_element_type(data)
                data_array = common_utils.convert_list_to_array(utils.utils.flatten_list(data), dtype=self.dtype)
            self.shape = tuple(utils.utils.get_shape(data))
            self.data = memref.create_memref(
                shape=self.shape,
                dtype=self.dtype,
                array=data_array,
            )
            self.device = utils.utils.default(device, tp_device.create_directly("gpu", 0))

        # Set data_str only for objects that won't be treated as Trace inputs
        if not utils.utils.should_lift_storage_op_as_input(self.shape):
            self.data_str = str(original_data)  # TODO (#448): Fix floating point str representation

        self.outputs[0].shape = list(self.shape)

    def str_skip_fields(self) -> Set[str]:
        # skip data since it is always a memref value
        return {"data"}

    def __eq__(self, other) -> bool:
        return self.data == other.data if isinstance(other, Storage) else False

    def infer_rank(self):
        # In the storage op, we actually know the exact shape, which we've already set in the constructor.
        # Hence, we don't need to set rank here (and doing so would overwrite the shape we set).
        pass

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        # TODO(#155): Fix allocation on host
        self.outputs[0].device = tp_device.create_directly("gpu", 0)

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import ConstantOp

        ConstantOp.build(inputs, outputs, data=self.data)
