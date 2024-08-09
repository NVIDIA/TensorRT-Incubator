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

import tripy.common
from tripy import utils
from tripy.common import utils as common_utils
from tripy.common.array import Array
from tripy.common.exception import raise_error
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Storage(BaseTraceOp):

    data: Union[Array, Sequence]
    shape: Sequence[int]
    dtype: type
    device: tripy.common.device

    def __init__(
        self,
        inputs: List["Tensor"],
        outputs: List["Tensor"],
        data: Union[Array, Sequence],
        shape: Sequence[int] = None,
        dtype: "tripy.dtype" = None,
        device: tripy.common.device = None,
    ) -> None:
        super().__init__(inputs, outputs)

        self.data = data
        self.has_array = isinstance(data, Array)
        if self.has_array:
            self.shape = data.shape
            self.dtype = data.dtype
            self.device = data.device
        else:
            data_shape = tuple(utils.get_shape(data))
            if shape is not None and tuple(shape) != data_shape:
                raise_error(
                    "Data has incorrect shape.",
                    details=[
                        f"Input data had shape: {data_shape}, ",
                        f"but provided shape was: {shape}",
                    ],
                )
            self.shape = data_shape
            if dtype is None:
                data_dtype = common_utils.get_element_type(data)
                dtype = common_utils.convert_frontend_dtype_to_tripy_dtype(data_dtype)
            self.dtype = dtype
            self.device = utils.default(device, tripy.common.device("gpu"))

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
        # This is different from self.device
        # Constants are always on device when executed by mlir
        self.outputs[0].device = tripy.common.device("gpu")

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConstantOp

        ConstantOp.build(inputs, outputs, data=self.data)
