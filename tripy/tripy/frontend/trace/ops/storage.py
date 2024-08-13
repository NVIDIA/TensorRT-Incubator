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

from dataclasses import dataclass
from typing import List, Sequence, Set, Union

from tripy import utils
from tripy.backend.mlir import utils as mlir_utils
from tripy.common import utils as common_utils
from tripy.common import device as tp_device
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp

import mlir_tensorrt.runtime.api as runtime

@dataclass(repr=False)
class Storage(BaseTraceOp):

    data: Union["runtime.MemRefValue", Sequence]
    shape: Sequence[int]
    dtype: type
    device: tp_device

    def __init__(
        self,
        inputs: List["Tensor"],
        outputs: List["Tensor"],
        data: Union["runtime.MemRefValue", Sequence],
        dtype: "tripy.dtype" = None,
        device: "tripy.device" = None,
    ) -> None:
        super().__init__(inputs, outputs)

        self.data = data
        self.dtype = dtype
        self.device = utils.default(device, tp_device("gpu"))
        if isinstance(data, Sequence):
            if self.dtype is None:
                self.dtype = common_utils.get_element_type(data)
            self.shape = utils.get_shape(data)
            self.has_memref = False
        else:
            if self.dtype is None:
                self.dtype = mlir_utils.convert_runtime_dtype_to_tripy_dtype(self.data.dtype)
            self.shape = data.shape
            self.has_memref = True

    # for storage, we will always consider the result to be an ordinary tensor
    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def str_skip_fields(self) -> Set[str]:
        if utils.should_omit_constant_in_str(self.shape):
            return {"data"}
        return set()

    def __eq__(self, other) -> bool:
        return self.data == other.data if isinstance(other, Storage) else False

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_rank(self):
        self.outputs[0].rank = len(self.shape)
        self.outputs[0].shape = self.shape

    def infer_devices(self):
        self.outputs[0].device = self.device

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.flat_ir.ops import ConstantOp, ConvertOp

        if self.has_memref and self.dtype != mlir_utils.convert_runtime_dtype_to_tripy_dtype(self.data.dtype):
            cast_tensor = FlatIRTensor.build(
                shape=outputs[0].shape,
                rank=outputs[0].rank,
                dtype=mlir_utils.convert_runtime_dtype_to_tripy_dtype(self.data.dtype),
                device=outputs[0].device,
                reason_details=["Cast constant tensor to target dtype"],
            )
            ConstantOp.build(inputs, [cast_tensor], data=self.data)
            ConvertOp.build([cast_tensor], outputs)
        else:
            ConstantOp.build(inputs, outputs, data=self.data)
