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

from tripy import utils
from tripy.backend.mlir import memref
from tripy.backend.mlir import utils as mlir_utils
from tripy.common import datatype
from tripy.common import device as tp_device
from tripy.common import utils as common_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


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
        is_memref = isinstance(data, runtime.MemRefValue)

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
                data_array = common_utils.convert_list_to_array(utils.flatten_list(data), dtype=self.dtype)
            self.shape = tuple(utils.get_shape(data))
            self.data = memref.create_memref(
                shape=self.shape,
                dtype=self.dtype,
                array=data_array,
            )
            self.device = utils.default(device, tp_device.create_directly("gpu", 0))

        # Set data_str only for non memref original data objects that won't be treated as Trace inputs
        if not is_memref and not utils.should_lift_storage_op_as_input(self.shape):
            self.data_str = str(original_data)  # TODO: add comment about fp cache hits

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
        from tripy.flat_ir.ops import ConstantOp

        ConstantOp.build(inputs, outputs, data=self.data)
