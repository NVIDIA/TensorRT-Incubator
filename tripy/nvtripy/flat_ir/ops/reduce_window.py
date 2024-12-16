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
from typing import Sequence, Tuple

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from nvtripy.backend.mlir import utils as mlir_utils
from nvtripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ReduceWindowOp(BaseFlatIROp):

    reduce_mode: str
    window_dims: Sequence[int]
    window_strides: Sequence[int]
    padding: Sequence[Tuple[int]]

    def _get_reduce_func(self):
        if self.reduce_mode == "max":
            return stablehlo.MaxOp
        elif self.reduce_mode == "avg":
            return stablehlo.AddOp
        else:
            raise NotImplementedError()

    def to_mlir(self, operands):
        input_dtype = self.inputs[0].dtype
        out_type = self.outputs[0].to_mlir()

        window_dims_attr = ir.DenseI64ArrayAttr.get(self.window_dims)
        window_strides_attr = ir.DenseI64ArrayAttr.get(self.window_strides)
        padding_attr_type = ir.RankedTensorType.get(
            [len(self.padding), 2],
            ir.IntegerType.get_signless(64),
        )
        padding_attr = ir.DenseElementsAttr.get(
            attrs=mlir_utils.list_to_dense_attr(self.padding, ir.IntegerType.get_signless(64)),
            type=padding_attr_type,
        )

        reduce = stablehlo.ReduceWindowOp(
            result=[out_type],
            inputs=[operands[0]],
            init_values=[operands[1]],
            window_dimensions=window_dims_attr,
            window_strides=window_strides_attr,
            padding=padding_attr,
        )

        reduce_arg_type = ir.RankedTensorType.get(
            [],
            mlir_utils.get_mlir_dtype(input_dtype),
        )
        reduce_block = ir.Block.create_at_start(reduce.regions[0], [reduce_arg_type, reduce_arg_type])
        reduce_func = self._get_reduce_func()
        with ir.InsertionPoint(reduce_block):
            out = reduce_func(*reduce_block.arguments)
            stablehlo.ReturnOp([out])

        return [reduce]
