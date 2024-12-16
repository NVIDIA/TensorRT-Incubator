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
from typing import List

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from nvtripy.backend.mlir.utils import get_mlir_dtype
from nvtripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ReduceOp(BaseFlatIROp):

    reduce_mode: str
    reduce_dims: List[int]

    # TODO(#87): Reuse flat ir ops
    def _get_reduce_func(self):
        if self.reduce_mode == "sum":
            return stablehlo.AddOp
        elif self.reduce_mode == "max":
            return stablehlo.MaxOp
        elif self.reduce_mode == "mul":
            return stablehlo.MulOp
        elif self.reduce_mode == "and":
            return stablehlo.AndOp
        elif self.reduce_mode == "or":
            return stablehlo.OrOp
        else:
            raise NotImplementedError()

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        dims_attr = ir.DenseI64ArrayAttr.get(self.reduce_dims)
        reduce = stablehlo.ReduceOp(
            result=[out_type],
            inputs=[operands[0]],
            init_values=[operands[1]],
            dimensions=dims_attr,
        )

        input_dtype = get_mlir_dtype(self.inputs[0].dtype)
        reduce_arg_type = ir.RankedTensorType.get(
            [],
            input_dtype,
        )
        reduce_block = ir.Block.create_at_start(reduce.regions[0], [reduce_arg_type, reduce_arg_type])
        reduce_func = self._get_reduce_func()
        with ir.InsertionPoint(reduce_block):
            out = reduce_func(*reduce_block.arguments)
            stablehlo.ReturnOp([out])

        return [reduce]


@dataclass(repr=False)
class ArgMinMaxOp(ReduceOp):
    """
    Operation for argmin and argmax.
    """

    # TODO: wrap the reducer in a func.call
    def _reducer(self, args):
        lhs_val, lhs_idx = args[0], args[1]
        rhs_val, rhs_idx = args[2], args[3]

        compare_val_dir = "GE" if self.reduce_mode == "argmax" else "LE"
        compare_val_dir = stablehlo.ComparisonDirectionAttr.get(compare_val_dir)
        eq_compare_dir = stablehlo.ComparisonDirectionAttr.get("EQ")

        compare_val = stablehlo.CompareOp(lhs_val, rhs_val, compare_val_dir)
        select_val = stablehlo.SelectOp(compare_val, lhs_val, rhs_val)
        eq_compare = stablehlo.CompareOp(lhs_val, rhs_val, eq_compare_dir)
        eq_branch = stablehlo.MinOp(lhs_idx, rhs_idx)
        non_eq_branch = stablehlo.SelectOp(compare_val, lhs_idx, rhs_idx)
        select_idx = stablehlo.SelectOp(eq_compare, eq_branch, non_eq_branch)
        return [select_val, select_idx]

    def to_mlir(self, operands):
        out_idx_type = self.outputs[0].to_mlir()
        out_val_type = ir.RankedTensorType.get(
            [ir.ShapedType.get_dynamic_size() for s in range(self.outputs[0].rank)],
            get_mlir_dtype(self.inputs[0].dtype),
        )
        dims_attr = ir.DenseI64ArrayAttr.get(self.reduce_dims)
        reduce = stablehlo.ReduceOp(
            result=[out_val_type, out_idx_type],
            inputs=operands[0:2],
            init_values=operands[2:4],
            dimensions=dims_attr,
        )

        val_dtype = get_mlir_dtype(self.inputs[0].dtype)
        idx_dtype = get_mlir_dtype(self.inputs[1].dtype)
        reduce_val_type = ir.RankedTensorType.get([], val_dtype)
        reduce_idx_type = ir.RankedTensorType.get([], idx_dtype)
        # [lhs_val, lhs_idx, rhs_val, rhs_idx]
        reduce_arg_types = [reduce_val_type, reduce_idx_type] * 2
        reduce_block = ir.Block.create_at_start(reduce.regions[0], reduce_arg_types)
        with ir.InsertionPoint(reduce_block):
            outs = self._reducer(reduce_block.arguments)
            stablehlo.ReturnOp(outs)
        out = reduce.results[1]
        return [out]
