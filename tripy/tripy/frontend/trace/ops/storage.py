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
from typing import List, Sequence, Set, Union

from tripy import utils
from tripy.backend.mlir import memref
from tripy.backend.mlir import utils as mlir_utils
from tripy.common import datatype
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
        dtype: datatype = None,
        device: tp_device = None,
    ) -> None:
        super().__init__(inputs, outputs)

        self.data = data
        if isinstance(data, runtime.MemRefValue):
            assert not any([dtype, device]), "Internal usage: dtype/device are inherited from memref."
            self.dtype = mlir_utils.convert_runtime_dtype_to_tripy_dtype(self.data.dtype)
            self.shape = tuple(data.shape)
            self.device = tp_device("gpu") if data.address_space == runtime.PointerType.device else tp_device("cpu")
            self.has_memref = True
        elif common_utils.is_empty(data):
            # special case: empty tensor
            self.dtype = utils.default(dtype, datatype.float32)
            self.shape = tuple(utils.get_shape(data))
            self.data = memref.create_memref(shape=self.shape, dtype=self.dtype)
            self.device = utils.default(device, tp_device("gpu"))
            self.has_memref = True
        else:
            self.dtype = dtype if dtype else common_utils.get_element_type(data)
            self.shape = tuple(utils.get_shape(data))
            self.device = utils.default(device, tp_device("gpu"))
            self.has_memref = False

    # for storage, we will always consider the result to be an ordinary tensor
    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def str_skip_fields(self) -> Set[str]:
        if utils.should_omit_constant_in_str(self.shape):
            return {"data"}
        return set()

    def __str__(self) -> str:
        skip_fields = self.str_skip_fields()
        args = []
        if "data" not in skip_fields:
            data_str = memref.pretty_print_memref(self.data) if self.has_memref else str(self.data)
            args.append(f"data={data_str}")
        args.extend([f"{field}={getattr(self, field)}" for field in ("shape", "dtype", "device")])
        return f"{self.outputs[0].name} = storage({', '.join([inp.name for inp in self.inputs] + args)})"

    def __eq__(self, other) -> bool:
        return self.data == other.data if isinstance(other, Storage) else False

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_rank(self):
        self.outputs[0].rank = len(self.shape)
        self.outputs[0].shape = self.shape

    def infer_devices(self):
        # TODO(#155): Fix allocation on host
        self.outputs[0].device = tp_device("gpu")

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConstantOp

        ConstantOp.build(inputs, outputs, data=self.data)
