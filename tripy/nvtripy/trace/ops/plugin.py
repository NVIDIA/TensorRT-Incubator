#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import numbers
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import mlir_tensorrt.compiler.api as compiler
from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy import utils
from nvtripy.common.exception import raise_error
from nvtripy.trace.ops.base import TraceOp
from nvtripy.utils.result import Result


@utils.utils.call_once
def initialize_plugin_registry():
    import tensorrt as trt

    major_version, _, _ = trt.__version__.partition(".")

    # TODO (#191): Make this work on Windows too
    handle = ctypes.CDLL(f"libnvinfer_plugin.so.{major_version}")
    handle.initLibNvInferPlugins(None, "")


def plugin_field_to_attr(field_info: "compiler.PluginFieldInfo", values: Any) -> Result[Any]:
    values = utils.utils.make_list(values)

    if field_info.type == compiler.PluginFieldType.CHAR:
        if len(values) != 1:
            return Result.err(
                [f"Expected a single string."],
            )
        if not isinstance(values[0], str):
            return Result.err([f"Expected a string."])
        return Result.ok(ir.StringAttr.get(values[0]))
    elif field_info.type == compiler.PluginFieldType.DIMS:
        if values and not isinstance(values[0], Sequence):
            values = [values]

        if any(not isinstance(value, Sequence) for value in values) or any(
            not isinstance(elem, int) for value in values for elem in value
        ):
            return Result.err([f"Expected sequences of integer values."])

        return Result.ok(ir.ArrayAttr.get([ir.DenseI64ArrayAttr.get(value) for value in values]))
    elif field_info.type == compiler.PluginFieldType.UNKNOWN:
        raise NotImplementedError("This function cannot currently handle unknown plugin field types")

    FLOAT_TYPES = {
        compiler.PluginFieldType.FLOAT64: ir.F64Type.get,
        compiler.PluginFieldType.FLOAT32: ir.F32Type.get,
        compiler.PluginFieldType.FLOAT16: ir.F16Type.get,
        compiler.PluginFieldType.BF16: ir.BF16Type.get,
        compiler.PluginFieldType.FP8: ir.Float8E4M3FNType.get,
    }

    INT_TYPES = {
        compiler.PluginFieldType.INT64: lambda: ir.IntegerType.get_signless(64),
        compiler.PluginFieldType.INT32: lambda: ir.IntegerType.get_signless(32),
        compiler.PluginFieldType.INT16: lambda: ir.IntegerType.get_signless(16),
        compiler.PluginFieldType.INT8: lambda: ir.IntegerType.get_signless(8),
    }
    try:
        compiler.PluginFieldType.INT4
    except AttributeError:
        pass
    else:
        INT_TYPES[compiler.PluginFieldType.INT4] = lambda: ir.IntegerType.get_signless(4)

    if field_info.type in FLOAT_TYPES:
        if any(not isinstance(value, numbers.Number) for value in values):
            return Result.err([f"Expected number(s)."])
        attrs = [ir.FloatAttr.get(FLOAT_TYPES[field_info.type](), value) for value in values]

    elif field_info.type in INT_TYPES:
        if any(not isinstance(value, int) for value in values):
            return Result.err([f"Expected integer(s)."])
        attrs = [ir.IntegerAttr.get(INT_TYPES[field_info.type](), value) for value in values]

    if field_info.length > 0 and len(attrs) != field_info.length:
        return Result.err([f"Expected {field_info.length} value(s), but got {len(attrs)}."])

    if len(attrs) > 1:
        return Result.ok(ir.DenseElementsAttr.get(attrs))
    return Result.ok(attrs[0])


@dataclass(repr=False)
class Plugin(TraceOp):
    name: str
    version: str
    namespace: str
    output_info: List[Tuple[int, "nvtripy.dtype"]]
    creator_params: Dict[str, Any]

    def __post_init__(self):
        super().__post_init__()
        self._originating_expression = self.outputs[0]

    def get_num_outputs(self):
        return len(self.output_info)

    def infer_dtypes(self):
        for out, (_, dtype) in zip(self.outputs, self.output_info):
            out.dtype = dtype

    def infer_rank(self):
        for out, (rank, _) in zip(self.outputs, self.output_info):
            out.rank = rank

    def to_mlir(self, inputs, outputs):
        initialize_plugin_registry()

        field_schema = compiler.get_tensorrt_plugin_field_schema(self.name, self.version, self.namespace, "")

        plugin_err_prefix = f"Plugin: {self.name} (version={self.version}, namespace={repr(self.namespace)})"

        params = {}
        for name, values in self.creator_params.items():
            if name not in field_schema:
                raise_error(
                    f"{plugin_err_prefix} has no field called: {name}",
                    [
                        f"Note: Valid fields are: {list(sorted(field_schema.keys()))}\n\n",
                        "This originated from:",
                        self._originating_expression,
                    ],
                )

            result = plugin_field_to_attr(field_schema[name], values)
            if not result:
                raise_error(
                    f"{plugin_err_prefix}: Invalid value provided for field: {name}",
                    result.error_details
                    + [f" Note: Provided field value was: {repr(values)}\n\n"]
                    + ["This originated from:", self._originating_expression],
                )
            params[name] = result.value

        return [
            tensorrt.opaque_plugin(outputs, self.name, self.version, self.namespace, ir.DictAttr.get(params), inputs)
        ]
