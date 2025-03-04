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

from typing import Dict, List, Sequence, Set

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import func as func_dialect
from mlir_tensorrt.compiler.dialects._ods_common import get_op_result_or_value
from nvtripy import utils
from nvtripy.backend.mlir.utils import (
    make_ir_context,
    make_tensor_location,
    map_error_to_user_code_and_raise,
    redirect_stderr,
)
from nvtripy.common.exception import raise_error
from nvtripy.common.shape_bounds import ShapeBounds
from nvtripy.logging import logger
from nvtripy.trace.ops.constant import Constant
from nvtripy.trace.tensor import TraceTensor
from nvtripy.trace.utils import topological_sort


class Trace:
    """
    A flattened representation of a computation graph expressed by one or more Tensors.
    """

    def __init__(
        self,
        outputs: Sequence[TraceTensor],
        inputs: Sequence[TraceTensor] = [],
        shapes: Sequence[ShapeBounds] = None,
        name: str = "main",
    ) -> None:
        """
        Args:
            tensors: The output TraceTensor(s) to evaluate.
            inputs: Input TraceTensor(s).
            shapes: The shape profile, consisting of min, opt, and max shapes for each input tensor.
                    Must be in the same order as `inputs`.
        """
        self.ops: List["BaseTraceOp"] = []
        self.inputs: List[TraceTensor] = inputs
        self.outputs: List[TraceTensor] = outputs
        self.shapes = shapes
        self.name = name

        exprs = [tensor.producer for tensor in outputs]

        input_op_ids = set(id(inp.producer) for inp in inputs)
        seen_op_ids: Set[int] = set()

        # Check all tensors for duplicate names. We currently rely on tensor names being
        # unique in the trace. We could potentially change this in the future to
        # automatically make names unique instead of complaining to the user, but it's better
        # for traceability if we use the names set by the user/frontend.
        self.tensor_map = {}

        def check_name(tensor):
            if tensor.name in self.tensor_map and (self.tensor_map[tensor.name] is not tensor):
                raise_error(
                    f"Found distinct tensors with the same name: '{tensor.name}'.",
                    details=["Tensor: ", tensor, "has the same name as another tensor: ", self.tensor_map[tensor.name]],
                )
            self.tensor_map[tensor.name] = tensor

        while exprs:
            head = exprs.pop(0)

            if id(head) in seen_op_ids:
                continue
            seen_op_ids.add(id(head))

            for io in head.inputs + head.outputs:
                check_name(io)

            if id(head) not in input_op_ids:
                # not as an input
                self.ops.append(head)
                exprs.extend([inp.producer for inp in head.inputs])

        # Reverse the order of the layers so they are topologically sorted
        self.ops = topological_sort(self.ops)

        logger.trace(lambda: f"{self}\n")

    def __str__(self) -> str:
        layer_strs: List[str] = []
        if self.shapes:
            layer_strs.append("input shapes:")
            for shape in self.shapes:
                layer_strs.append(f"    {str(shape)}")

        if len(self.inputs):
            layer_strs.append("inputs:")
        for inp in self.inputs:
            layer_strs.append(f"    {str(inp)}")
        for op in self.ops:
            layer_strs.append(str(op))
        layer_strs.append("outputs:")
        for out in self.outputs:
            layer_strs.append(f"    {str(out)}")
        return "\n".join(layer_strs)

    @staticmethod
    def _collect_constant_tensors(trace_tensor):
        visited = set()
        inputs = []

        def dfs(trace_tensor):
            if id(trace_tensor) in visited:
                return
            visited.add(id(trace_tensor))

            producer = trace_tensor.producer
            if isinstance(producer, Constant) and utils.utils.should_lift_constant_op_as_input(producer.shape):
                inputs.append(trace_tensor)
            else:
                for inp in producer.inputs:
                    dfs(inp)

        dfs(trace_tensor)
        return inputs

    def to_mlir(self):
        def to_mlir_impl():

            with make_ir_context(), ir.Location.unknown():
                module = ir.Module.create()
                with ir.InsertionPoint(module.body) as ip:
                    func_op = func_dialect.FuncOp(
                        self.name,
                        ir.FunctionType.get(
                            [inp.to_mlir() for inp in self.inputs],
                            [out.to_mlir() for out in self.outputs],
                        ),
                        ip=ip,
                    )

                    entry_block = func_op.add_entry_block()
                    with ir.InsertionPoint(entry_block):
                        mlir_ops: Dict[str, ir.BlockArgument] = {}
                        # Initialize tensor dict with inputs
                        for index, inp in enumerate(self.inputs):
                            mlir_ops[inp.name] = entry_block.arguments[index]

                        for op in self.ops:
                            layer_inputs = [mlir_ops[inp.name] for inp in op.inputs]
                            layer_outputs = [out.to_mlir() for out in op.outputs]

                            with make_tensor_location(
                                [inp.name for inp in op.inputs], [out.name for out in op.outputs]
                            ):
                                layer_outputs = op.to_mlir(layer_inputs, layer_outputs)

                                # TODO (pranavm): Check if this is needed:
                                # stablehlo python bindings can do some naive shape and type inference.
                                # If the shapes are frozen after adding a layer, assign these shapes back to trace tensor.
                                for mlir_out, trace_out in zip(layer_outputs, op.outputs):
                                    type = get_op_result_or_value(mlir_out).type
                                    trace_out.shape = tuple(
                                        (-1 if type.is_dynamic_dim(i) else type.get_dim_size(i))
                                        for i in range(type.rank)
                                    )

                            mlir_ops.update(zip([out.name for out in op.outputs], layer_outputs))

                        func_dialect.ReturnOp([mlir_ops[o.name] for o in self.outputs])

                    # TODO (pranavm): Check if this is needed:
                    # After lowering the complete graph to stablehlo, there can be mismatch between Tripy created function signature and the ReturnOp due to shapes that resolved while lowering into Stablehlo.
                    # Here, we check if the types for the results and change the function signature to obey the inferred types.
                    new_out_types = [get_op_result_or_value(mlir_ops[o.name]).type for o in self.outputs]
                    ftype = ir.FunctionType.get([inp.to_mlir() for inp in self.inputs], new_out_types)
                    func_op.attributes["function_type"] = ir.TypeAttr.get(ftype)

                    if self.shapes:
                        # Create tensorrt.shape_profile attribute for all function arguments
                        arg_attrs: List[Dict[str, ir.Attribute]] = []
                        for bound in self.shapes:
                            # TODO (#244): Support multiple profiles
                            arg_attrs.append(
                                ir.DictAttr.get(
                                    {
                                        "tensorrt.shape_profile": ir.Attribute.parse(
                                            f"#tensorrt.shape_profile<min={list(bound.min)}, opt={list(bound.opt)}, max={list(bound.max)}>"
                                        )
                                    }
                                )
                            )
                        func_op.arg_attrs = ir.ArrayAttr.get(arg_attrs)

                    # Append device location if outputs are on host as MLIR-TensorRT does not adhere to this constraint.
                    # TODO(#155): Fix TensorKindAnalysis to ensure result tensors with attribute `tensorrt.host_tensor` are allocated on host.
                    res_attrs = []
                    for output in self.outputs:
                        if output.device.kind == "cpu":
                            res_attrs.append(ir.Attribute.parse("{tensorrt.host_tensor}"))
                        else:
                            res_attrs.append(ir.DictAttr.get({}))
                    func_op.res_attrs = ir.ArrayAttr.get(res_attrs)

                module.operation.attributes["sym_name"] = ir.StringAttr.get(
                    utils.utils.UniqueNameGen.gen_uid(
                        [inp.name for inp in self.inputs], [out.name for out in self.outputs]
                    )
                )
                return module

        try:
            with redirect_stderr() as outfile:
                mlir = to_mlir_impl()
        except Exception as exc:

            outfile.flush()
            outfile.seek(0)
            stderr = outfile.read()

            map_error_to_user_code_and_raise(self, exc, stderr.decode())

        return mlir
