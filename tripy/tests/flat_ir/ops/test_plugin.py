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

import mlir_tensorrt.compiler.api as compiler
import pytest
from mlir_tensorrt.compiler import ir

import tripy as tp
from tripy.backend.mlir import utils as mlir_utils
from tripy.flat_ir.ops import PluginOp
from tripy.flat_ir.ops.plugin import plugin_field_to_attr
from tripy.frontend.trace import Trace
from typing import Sequence
from tests import helper


@pytest.fixture
def flat_ir():
    X = tp.iota((1, 2, 4, 4))
    X.name = "X"
    rois = tp.Tensor([[0.0, 0.0, 9.0, 9.0], [0.0, 5.0, 4.0, 9.0]], dtype=tp.float32, name="rois")
    batch_indices = tp.zeros((2,), dtype=tp.int32)
    batch_indices.name = "batch_indices"

    out = tp.plugin(
        "ROIAlign_TRT", [X, rois, batch_indices], output_info=[(X.rank, X.dtype)], output_height=5, output_width=5
    )
    out.name = "out"

    trace = Trace([out])
    flat_ir = trace.to_flat_ir()
    yield flat_ir


class TestPluginOp:
    def test_str(self, flat_ir):
        plugin_op = flat_ir.ops[-1]
        assert isinstance(plugin_op, PluginOp)
        assert (
            str(plugin_op)
            == "out: [rank=(4), dtype=(float32), loc=(gpu:0)] = PluginOp(X, rois, batch_indices, name='ROIAlign_TRT', version='1', namespace='', creator_params={'output_height': 5, 'output_width': 5})"
        )

    def test_mlir(self, flat_ir):
        assert """
            tensorrt.opaque_plugin {creator_params = {output_height = 5 : i32, output_width = 5 : i32}, plugin_name = "ROIAlign_TRT", plugin_namespace = "", plugin_version = "1"}(%0, %cst, %2) : (tensor<?x?x?x?xf32>, tensor<2x4xf32>, tensor<?xi32>) -> tensor<?x?x?x?xf32>
            """.strip() in str(
            flat_ir.to_mlir()
        )

    @pytest.mark.parametrize(
        "params, expected_error",
        [
            # Nonexistent parameter
            (
                {"not_a_real_param": 1},
                r"has no field called: not_a_real_param\n\s*Note: Valid fields are:",
            ),
            # Incorrect field type
            (
                {"output_height": 5.0},
                r"Invalid value provided for field: output_height\n\s*Expected integer\(s\). Note: Provided field value was: 5.0",
            ),
            # Incorrect number of items
            (
                {"output_height": [5, 5]},
                r"Invalid value provided for field: output_height\n\s*Expected 1 value\(s\), but got 2. Note: Provided field value was: \[5, 5\]",
            ),
        ],
    )
    def test_incorrect_plugin_fields(self, params, expected_error):
        X = tp.iota((1, 2, 4, 4))
        rois = tp.Tensor([[0.0, 0.0, 9.0, 9.0], [0.0, 5.0, 4.0, 9.0]], dtype=tp.float32)
        batch_indices = tp.zeros((2,), dtype=tp.int32)

        out = tp.plugin(
            "ROIAlign_TRT",
            [X, rois, batch_indices],
            output_info=[(X.rank, X.dtype)],
            **params,
        )

        with helper.raises(tp.TripyException, expected_error, has_stack_info_for=[out]):
            out.eval()


@dataclass
class FieldInfo:
    type: compiler.PluginFieldType
    length: int


class TestPluginFieldToAttr:
    def has_error(self, result, expected_error):
        if expected_error:
            assert not result
            assert expected_error in " ".join(result.error_details)
            return True
        else:
            assert result
            return False

    @pytest.mark.parametrize(
        "inp,expected_error",
        [
            ("test_string", None),
            (["test_string", "other_string"], "Expected a single string"),
            (1, "Expected a string"),
        ],
    )
    def test_str(self, inp, expected_error):
        with mlir_utils.make_ir_context():
            result = plugin_field_to_attr(
                FieldInfo(compiler.PluginFieldType.CHAR, len(inp) if isinstance(inp, str) else 1), inp
            )
            if not self.has_error(result, expected_error):
                attr = result.value
                assert isinstance(attr, ir.StringAttr)
                assert attr.value == inp

    @pytest.mark.parametrize(
        "inp, expected_error",
        [
            ((1, 2, 3), None),
            ("hello", "Expected sequences of integer values"),
            ((1.5, 2.6, 3.7), "Expected sequences of integer values"),
        ],
    )
    def test_dims(self, inp, expected_error):
        def to_list(value):
            if not isinstance(value[0], Sequence):
                return [value]
            return value

        with mlir_utils.make_ir_context():
            result = plugin_field_to_attr(FieldInfo(compiler.PluginFieldType.DIMS, len(inp)), inp)
            if not self.has_error(result, expected_error):
                attr = result.value
                assert isinstance(attr, ir.ArrayAttr)
                assert all(isinstance(shape, ir.DenseI64ArrayAttr) for shape in attr)
                assert all(list(shape) == list(inp_shape) for shape, inp_shape in zip(attr, to_list(inp)))

    @pytest.mark.parametrize(
        "inp, expected_error",
        [
            (1, None),
            (1.0, None),
            ([1.0, 1], None),
            ("hello", "Expected number(s)"),
        ],
    )
    @pytest.mark.parametrize(
        "plugin_field_type,expected_type",
        [
            (compiler.PluginFieldType.FLOAT64, ir.F64Type),
            (compiler.PluginFieldType.FLOAT32, ir.F32Type),
            (compiler.PluginFieldType.FLOAT16, ir.F16Type),
            (compiler.PluginFieldType.BF16, ir.BF16Type),
            (compiler.PluginFieldType.FP8, ir.Float8E4M3FNType),
        ],
    )
    def test_float(self, inp, expected_error, plugin_field_type, expected_type):
        with mlir_utils.make_ir_context(), ir.Location.unknown():
            num_values = len(inp) if isinstance(inp, Sequence) else 1
            result = plugin_field_to_attr(FieldInfo(plugin_field_type, num_values), inp)
            if not self.has_error(result, expected_error):
                attr = result.value
                if num_values == 1:
                    assert isinstance(attr, ir.FloatAttr)
                    assert attr.value == inp
                    assert attr.type == expected_type.get()
                else:
                    assert isinstance(attr, ir.DenseElementsAttr)
                    assert len(attr) == num_values

    @pytest.mark.parametrize(
        "inp, expected_error",
        [
            (1, None),
            ([1, 2], None),
            (1.0, "Expected integer(s)"),
            ("hello", "Expected integer(s)"),
        ],
    )
    @pytest.mark.parametrize(
        "plugin_field_type,expected_type",
        [
            (compiler.PluginFieldType.INT64, lambda: ir.IntegerType.get_signless(64)),
            (compiler.PluginFieldType.INT32, lambda: ir.IntegerType.get_signless(32)),
            (compiler.PluginFieldType.INT16, lambda: ir.IntegerType.get_signless(16)),
            (compiler.PluginFieldType.INT8, lambda: ir.IntegerType.get_signless(8)),
        ],
    )
    def test_int(self, inp, expected_error, plugin_field_type, expected_type):
        with mlir_utils.make_ir_context(), ir.Location.unknown():
            expected_type = expected_type()
            num_values = len(inp) if isinstance(inp, Sequence) else 1
            result = plugin_field_to_attr(FieldInfo(plugin_field_type, num_values), inp)
            if not self.has_error(result, expected_error):
                attr = result.value
                if num_values == 1:
                    assert isinstance(attr, ir.IntegerAttr)
                    assert attr.value == inp
                    assert attr.type == expected_type
                else:
                    assert isinstance(attr, ir.DenseElementsAttr)
                    assert len(attr) == num_values

    def test_incorrect_number_of_values(self):
        with mlir_utils.make_ir_context(), ir.Location.unknown():
            result = plugin_field_to_attr(FieldInfo(compiler.PluginFieldType.FLOAT32, 3), [1, 2])
            assert self.has_error(result, "Expected 3 value(s), but got 2.")
