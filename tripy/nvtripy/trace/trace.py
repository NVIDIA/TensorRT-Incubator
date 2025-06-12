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

from textwrap import indent
from typing import Dict, List, Optional, Sequence, Set

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
from nvtripy.logging import logger
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
        input_infos: Optional[Dict[str, "nvtripy.InputInfo"]] = None,
        name: str = "main",
    ) -> None:
        # ops/inputs/outputs are populated by `trace()`
        self.ops: List["TraceOp"] = []
        self.inputs: List[TraceTensor] = []
        self.outputs: List[TraceTensor] = []
        self.input_infos = input_infos
        self.name = name

        self.trace(outputs, inputs)

    # Performs the actual tracing to populate self.ops
    def trace(self, outputs, inputs):
        self.inputs = inputs
        self.outputs = outputs

        ops = []
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
                ops.append(head)
                exprs.extend([inp.producer for inp in head.inputs])

        # Reverse the order of the layers so they are topologically sorted
        self.ops = topological_sort(ops)

        logger.trace(lambda: f"{self}\n")

    def __str__(self) -> str:
        TAB = " " * 4

        layer_strs: List[str] = []

        signature = f"def {self.name}("

        def get_sep(lst):
            return f"\n{TAB}" if lst else ""

        inp_sep = get_sep(self.inputs)
        input_strs = []
        for inp, inp_shape in zip(
            self.inputs,
            ([info for info in self.input_infos.values()] if self.input_infos else [None] * len(self.inputs)),
        ):
            input_strs.append(f"{inp}{f' : {inp_shape}' if inp_shape else ''}")
        signature += inp_sep + f",{inp_sep}".join(input_strs)
        if self.inputs:
            signature += f"\n"

        out_sep = get_sep(self.outputs)
        signature += f") -> (" + out_sep + f",{out_sep}".join(str(out) for out in self.outputs) + f"\n):"

        layer_strs.append(signature)
        for op in self.ops:
            layer_strs.append(indent(str(op), prefix=TAB))
        layer_strs.append(indent(f"return {', '.join(out.name for out in self.outputs)}", TAB))

        return "\n".join(layer_strs)

    def to_mlir(self):
        def to_mlir_impl():

            with make_ir_context(), ir.Location.unknown():
                module = ir.Module.create()
                with ir.InsertionPoint(module.body) as ip:
                    func_op = func_dialect.FuncOp(
                        "main",
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
                            layer_input_ops = [mlir_ops[inp.name] for inp in op.inputs]
                            output_types = [out.to_mlir() for out in op.outputs]

                            with make_tensor_location(
                                [inp.name for inp in op.inputs], [out.name for out in op.outputs]
                            ):
                                mlir_output_ops = op.to_mlir(layer_input_ops, output_types)

                            # When the MLIR ranked tensor type generated by Tripy has more information
                            # than what's in the MLIR operation's type, update the operation's type.
                            # This is required so that we can compute the shape of shape tensors
                            # (e.g. in tensor_from_shape_like)
                            for output_type, mlir_output_op in zip(output_types, mlir_output_ops):

                                def num_known_dims(ranked_tensor_type):
                                    return sum(
                                        1
                                        for i in range(ranked_tensor_type.rank)
                                        if not ranked_tensor_type.is_dynamic_dim(i)
                                    )

                                if num_known_dims(output_type) >= num_known_dims(mlir_output_op.type):
                                    mlir_output_op.set_type(output_type)

                            mlir_ops.update(zip([out.name for out in op.outputs], mlir_output_ops))

                        func_dialect.ReturnOp([mlir_ops[o.name] for o in self.outputs])

                    # Some type refinement is done during lowering, so the tensor types of the function may change at this point.
                    # Here we update the function signature with the inferred types.
                    new_inp_types = [get_op_result_or_value(mlir_ops[inp.name]).type for inp in self.inputs]
                    new_out_types = [get_op_result_or_value(mlir_ops[out.name]).type for out in self.outputs]
                    func_op.attributes["function_type"] = ir.TypeAttr.get(
                        ir.FunctionType.get(new_inp_types, new_out_types)
                    )

                    arg_attrs: List[Dict[str, ir.Attribute]] = []
                    for inp in self.inputs:
                        attr = {}
                        if self.input_infos:
                            input_info = self.input_infos[inp.name]
                            shape_bounds = input_info.shape_bounds
                            attr["tensorrt.shape_profile"] = ir.Attribute.parse(
                                f"#tensorrt.shape_profile<min={list(shape_bounds.min)}, opt={list(shape_bounds.opt)}, max={list(shape_bounds.max)}>"
                            )
                            attr["tensorrt.dimension_names"] = ir.DictAttr.get(
                                {str(idx): ir.StringAttr.get(name) for idx, name in input_info.dimension_names.items()}
                            )

                        arg_attrs.append(ir.DictAttr.get(attr))

                    func_op.arg_attrs = ir.ArrayAttr.get(arg_attrs)

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
