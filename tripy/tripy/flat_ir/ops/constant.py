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
from typing import Sequence, Set, Union

import mlir_tensorrt.runtime.api as runtime
from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy import utils
from tripy.backend.mlir.memref import create_memref
from tripy.common import device
from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ConstantOp(BaseFlatIROp):

    data: Union[runtime.MemRefValue, Sequence]

    def str_skip_fields(self) -> Set[str]:
        data_shape = self.data.shape if isinstance(self.data, runtime.MemRefValue) else self.outputs[0].shape
        if utils.should_omit_constant_in_str(data_shape):
            return {"data"}
        return set()

    def to_mlir(self, operands):
        import array

        import tripy.common.datatype as datatype
        from tripy.backend.mlir import utils as mlir_utils

        # TODO(#189): Remove explicit copy to host for constants
        if isinstance(self.data, runtime.MemRefValue):
            runtime_client = mlir_utils.MLIRRuntimeClient()
            data_memref = self.data
            if data_memref.address_space == runtime.PointerType.device:
                data_memref = runtime_client.copy_to_host(
                    device_memref=data_memref,
                    stream=None,
                )

            # TODO: we can further drop the cast by tolist(memref) -> mlir
            # Workaround (#208): bools are represented as i1 in MLIR-TRT but they cannot be used for DenseElementsAttr
            # so we have to represent them as ints and then cast the result
            if self.outputs[0].dtype == datatype.bool:
                # need to use memoryview.cast to ensure that the view will be flattened
                int_memref = create_memref(
                    array=array.array("i", memoryview(data_memref).cast("b").tolist()),
                    shape=self.data.shape,
                    dtype=datatype.int32,
                    device=device("cpu"),
                )
                attr = ir.DenseElementsAttr.get(
                    array=int_memref, type=mlir_utils.get_mlir_dtype(datatype.int32), shape=data_memref.shape
                )
                cast_output = mlir_utils.make_mlir_tensor(datatype.bool, data_memref.shape)
                constant_op = stablehlo.ConstantOp(attr)
                return [stablehlo.ConvertOp(result=cast_output, operand=constant_op)]

            attr = ir.DenseElementsAttr.get(
                array=data_memref, type=mlir_utils.get_mlir_dtype(self.outputs[0].dtype), shape=data_memref.shape
            )
        else:
            out_dtype = self.outputs[0].dtype
            attr = ir.DenseElementsAttr.get(
                attrs=mlir_utils.list_to_dense_attr(self.data, mlir_utils.get_mlir_dtype(out_dtype)),
                type=self.outputs[0].to_mlir(),
            )

        return [stablehlo.ConstantOp(attr)]
