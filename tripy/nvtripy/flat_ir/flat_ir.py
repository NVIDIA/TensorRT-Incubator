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

from typing import Dict, List, Sequence, Set, Union

from mlir_tensorrt.compiler.dialects._ods_common import get_op_result_or_value
from mlir_tensorrt.runtime.api import MemRefValue

from nvtripy import utils
from nvtripy.common.shape_bounds import ShapeBounds
from nvtripy.utils.utils import list_to_tuple


class FlatIR:
    """
    A flattened low level representation of a computation graph which maps directly with StableHLO dialect.
    """

    def __init__(self, shapes: Sequence[ShapeBounds] = None):
        self.inputs: List["FlatIRTensor"] = []
        self.outputs: List["FlatIRTensor"] = []
        self.functions: Dict[str, "FlatIRFunction"] = {}
        self.ops: List[Union["BaseFlatIROp", "FlatIRFunction"]] = []

        self.shapes = shapes
        self.tensor_map: Dict[str] = {}

        self.tensor_replacements: Dict[int, "FlatIRTensor"] = {}
        self.constant_map = {}

    def __str__(self) -> str:
        """Generate a string representation of the FlatIR."""
        from nvtripy.flat_ir.function import FlatIRFunction
        from nvtripy.flat_ir.ops.base import BaseFlatIROp

        ir = []

        # Print functions
        for function in self.functions.values():
            ir.append(str(function))
            ir.append("")  # Empty line for readability

        ir.append("Main Function:")

        ir.append("inputs:")
        for input in self.inputs:
            ir.append(f"    {input}")

        for op in self.ops:
            if isinstance(op, FlatIRFunction):
                output_names = ", ".join(output.__str__() for output in op.get_caller_outputs())
                input_names = ", ".join(input.name for input in op.get_caller_inputs())
                ir.append(f"{output_names} = function {op.name}({input_names})")
            elif isinstance(op, BaseFlatIROp):
                ir.append(str(op))

        ir.append("outputs:")
        for output in self.outputs:
            ir.append(f"    {output}")

        return "\n".join(ir)

    def add_function(self, function: "FlatIRFunction") -> None:
        """
        Add a function to the FlatIR, ensuring unique function names.
        """
        base_function_name = function.name
        name_counter = 1

        # Ensure unique function names
        while function.name in self.functions:
            function.name = f"{base_function_name}_{name_counter}"
            name_counter += 1

        self.functions[function.name] = function

    def to_mlir(self):
        """
        Convert the FlatIR representation to MLIR.

        This method generates an MLIR module containing private functions for each FlatIRFunction
        and a main function that orchestrates the execution flow.
        """
        from mlir_tensorrt.compiler import ir
        from mlir_tensorrt.compiler.dialects import func as func_dialect

        from nvtripy.backend.mlir.utils import make_ir_context, make_tensor_location
        from nvtripy.flat_ir.function import FlatIRFunction
        from nvtripy.flat_ir.ops.base import BaseFlatIROp

        def _base_op_to_mlir(op, mlir_tensor_map):
            op_inputs = [mlir_tensor_map[input_tensor.name] for input_tensor in op.inputs]
            with make_tensor_location(
                [inp.name for inp in op.inputs],
                [out.name for out in op.outputs],
                op.trace_input_names,
                op.trace_output_names,
            ):
                op_outputs = op.to_mlir(op_inputs)

                # stablehlo python bindings can do some naive shape and type inference.
                # If the shapes are freezed after adding a layer, assign these shapes back to flat_ir tensor.
                for mlir_out, flatir_out in zip(op_outputs, op.outputs):
                    type = get_op_result_or_value(mlir_out).type
                    flatir_out.shape = tuple(
                        [(-1 if type.is_dynamic_dim(i) else type.get_dim_size(i)) for i in range(type.rank)]
                    )

                for mlir_output, flat_ir_output in zip(op_outputs, op.outputs):
                    mlir_tensor_map[flat_ir_output.name] = mlir_output

        def _func_op_to_mlir(
            insertion_point: ir.InsertionPoint,
            func: FlatIRFunction,
            mlir_tensor_map: Dict[str, ir.Value],
            mlir_func_map: Dict[str, ir.Operation],
        ) -> None:
            """
            Convert a FlatIRFunction to MLIR representation.
            """
            with make_tensor_location(
                [inp.name for inp in func.inputs],
                [out.name for out in func.outputs],
                func.trace_input_names,
                func.trace_output_names,
            ):
                if func.name not in mlir_func_map:
                    func_output_types = _create_new_function(insertion_point, func, mlir_tensor_map, mlir_func_map)
                else:
                    func_output_types = mlir_func_map[func.name].type.results

                _create_function_call(func, mlir_tensor_map, mlir_func_map, func_output_types)

        def _convert_to_dynamic_tensor(tensor) -> ir.RankedTensorType:
            from nvtripy.backend.mlir import utils as mlir_utils

            dynamic_shape = [ir.ShapedType.get_dynamic_size()] * tensor.rank
            return ir.RankedTensorType.get(dynamic_shape, mlir_utils.get_mlir_dtype(tensor.dtype))

        def _create_new_function(
            insertion_point: ir.InsertionPoint,
            func: FlatIRFunction,
            mlir_tensor_map: Dict[str, ir.Value],
            mlir_func_map: Dict[str, ir.Operation],
        ) -> [ir.Type]:
            """Create a new MLIR function for a FlatIRFunction."""
            func_input_types = _get_function_input_types(func, mlir_tensor_map)
            func_output_types = [_convert_to_dynamic_tensor(out_tensor) for out_tensor in func.get_caller_outputs()]
            func_type = ir.FunctionType.get(func_input_types, func_output_types)
            func_op = func_dialect.FuncOp(func.name, func_type, ip=insertion_point, visibility="private")

            with ir.InsertionPoint(func_op.add_entry_block()):
                _process_function_body(func, func_op, mlir_tensor_map)

            func_new_output_types = [get_op_result_or_value(mlir_tensor_map[o.name]).type for o in func.outputs]
            func_type = ir.FunctionType.get(func_input_types, func_new_output_types)
            func_op.attributes["function_type"] = ir.TypeAttr.get(func_type)

            mlir_func_map[func.name] = func_op

            return func_new_output_types

        def _get_function_input_types(func: FlatIRFunction, mlir_tensor_map: Dict[str, ir.Value]) -> List[ir.Type]:
            """Get the input types for a function, converting to dynamic tensors if necessary."""

            # Skip converting to dynamic tensor for Quantize/Dequantize scale operation.
            if "Quantize" in func.name or "Dequantize" in func.name:
                return [
                    _convert_to_dynamic_tensor(func.get_caller_inputs()[0]),
                    get_op_result_or_value(mlir_tensor_map[func.get_caller_inputs()[1].name]).type,
                ]
            else:
                return [_convert_to_dynamic_tensor(input_tensor) for input_tensor in func.get_caller_inputs()]

        def _process_function_body(
            func: FlatIRFunction, func_op: func_dialect.FuncOp, mlir_tensor_map: Dict[str, ir.Value]
        ) -> None:
            """Process the body of a function, converting each operation to MLIR."""
            for index, input_tensor in enumerate(func.inputs):
                mlir_tensor_map[input_tensor.name] = func_op.arguments[index]

            for op in func.ops:
                op_inputs = _get_op_inputs(op, mlir_tensor_map)
                op_outputs = op.to_mlir(op_inputs)

                for mlir_output, flat_ir_output in zip(op_outputs, op.outputs):
                    mlir_tensor_map[flat_ir_output.name] = mlir_output

            func_dialect.ReturnOp([mlir_tensor_map[output_tensor.name] for output_tensor in func.outputs])

        def _get_op_inputs(op: ir.Operation, mlir_tensor_map: Dict[str, ir.Value]) -> List[ir.Value]:
            """Get the inputs for an operation, casting to dynamic tensors if necessary."""
            from nvtripy.backend.mlir.utils import cast_to_dynamic_ranked_tensor, is_any_dim_dynamic
            from nvtripy.flat_ir.ops import DynamicBroadcastOp, DynamicReshapeOp

            op_inputs = []
            for index, op_input in enumerate(op.inputs):
                # Insert a cast operation between function input and reshape/broadcast op shape input.abs
                # `op.to_mlir` operation for such op, convert dynamic input to static due to stablehlo constraints.
                # Addition `cast` operation ensure that we decouple function input type with corresponding op input type.
                if (
                    (isinstance(op, (DynamicReshapeOp, DynamicBroadcastOp)))
                    and index == 1
                    and is_any_dim_dynamic(mlir_tensor_map[op_input.name])
                ):
                    op_inputs.append(
                        cast_to_dynamic_ranked_tensor(mlir_tensor_map[op_input.name], always_insert_cast=True)
                    )
                else:
                    op_inputs.append(mlir_tensor_map[op_input.name])
            return op_inputs

        def _create_function_call(
            func: FlatIRFunction,
            mlir_tensor_map: Dict[str, ir.Value],
            mlir_func_map: Dict[str, ir.Operation],
            func_output_types: List[ir.Type],
        ) -> None:
            """Create a function call operation in MLIR."""
            from nvtripy.backend.mlir.utils import cast_to_dynamic_ranked_tensor

            func_inputs = [mlir_tensor_map[input_tensor.name] for input_tensor in func.get_caller_inputs()]

            # Convert function input to dynamic so that it can be reused.
            # Skip converting to dynamic tensor for Quantize/Dequantize scale operation.
            if ("Quantize" in func.name or "Dequantize" in func.name) and len(func_inputs) > 1:
                dynamic_func_inputs = [cast_to_dynamic_ranked_tensor(func_inputs[0]), func_inputs[1]]
            else:
                dynamic_func_inputs = [cast_to_dynamic_ranked_tensor(input) for input in func_inputs]

            func_call_op = func_dialect.CallOp(
                func_output_types, ir.FlatSymbolRefAttr.get(func.name), dynamic_func_inputs
            )

            for index, output_tensor in enumerate(func.get_caller_outputs()):
                mlir_tensor_map[output_tensor.name] = func_call_op.results[index]

        def _update_main_function_signature(main_func_op, outputs, mlir_tensor_map, main_input_types):
            # After lowering the complete graph to stablehlo, there can be mismatch between Tripy created function signature and the ReturnOp due to shapes that resolved while lowering into Stablehlo.
            # Here, we check if the types for the results and change the function signature to obey the inferred types.
            man_new_out_types = [get_op_result_or_value(mlir_tensor_map[o.name]).type for o in outputs]
            main_func_type = ir.FunctionType.get(main_input_types, man_new_out_types)
            main_func_op.attributes["function_type"] = ir.TypeAttr.get(main_func_type)

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
                main_func_op.arg_attrs = ir.ArrayAttr.get(arg_attrs)

        def to_mlir_impl():
            with make_ir_context(), ir.Location.unknown():
                module = ir.Module.create()
                with ir.InsertionPoint(module.body) as insertion_point:
                    mlir_tensor_map: Dict[str, Union[ir.Operation, ir.Value]] = {}

                    # Create the main function
                    main_input_types = [input_tensor.to_mlir() for input_tensor in self.inputs]
                    main_output_types = [output_tensor.to_mlir() for output_tensor in self.outputs]
                    main_func_type = ir.FunctionType.get(main_input_types, main_output_types)
                    main_func_op = func_dialect.FuncOp("main", main_func_type, ip=insertion_point)

                    mlir_func_map: Dict[str, func_dialect.FuncOp] = {}

                    with ir.InsertionPoint(main_func_op.add_entry_block()):
                        # Map main function inputs to MLIR tensor map
                        for index, input_tensor in enumerate(self.inputs):
                            mlir_tensor_map[input_tensor.name] = main_func_op.arguments[index]

                        # Process each op in the main flow
                        for op in self.ops:
                            if isinstance(op, FlatIRFunction):
                                _func_op_to_mlir(insertion_point, op, mlir_tensor_map, mlir_func_map)
                            elif isinstance(op, BaseFlatIROp):
                                _base_op_to_mlir(op, mlir_tensor_map)

                        # Add return operation for the main function
                        func_dialect.ReturnOp([mlir_tensor_map[output_tensor.name] for output_tensor in self.outputs])

                    # Update main function signature if necessary
                    _update_main_function_signature(main_func_op, self.outputs, mlir_tensor_map, main_input_types)

                    # Append device location if outputs are on host as MLIR-TensorRT does not adhere to this constraint.
                    # TODO(#155): Fix TensorKindAnalysis to ensure result tensors with attribute `tensorrt.host_tensor` are allocated on host.
                    res_attrs = []
                    for output in self.outputs:
                        if output.device.kind == "cpu":
                            res_attrs.append(ir.Attribute.parse("{tensorrt.host_tensor}"))
                        else:
                            res_attrs.append(ir.DictAttr.get({}))
                    main_func_op.res_attrs = ir.ArrayAttr.get(res_attrs)

                module.operation.attributes["sym_name"] = ir.StringAttr.get(
                    utils.UniqueNameGen.gen_uid([inp.name for inp in self.inputs], [out.name for out in self.outputs])
                )

                return module

        from nvtripy.backend.mlir.utils import redirect_stderr

        try:
            with redirect_stderr() as outfile:
                mlir = to_mlir_impl()
        except Exception as exc:
            from nvtripy.backend.mlir.utils import map_error_to_user_code_and_raise

            outfile.flush()
            outfile.seek(0)
            stderr = outfile.read()

            map_error_to_user_code_and_raise(self, exc, stderr.decode())

        return mlir

    def register_tensor(self, tensor: "FlatIRTensor") -> "FlatIRTensor":
        """
        Registers a tensor with this FlatIR instance. If the tensor has no name, a name unique to this FlatIR will be assigned.
        """
        tensor.name = utils.default(tensor.name, f"t_inter{len(self.tensor_map)}")
        if tensor.name in self.tensor_map:
            return self.tensor_map[tensor.name]
        self.tensor_map[tensor.name] = tensor
        return tensor

    def _get_constant_key(self, op):
        if isinstance(op.data, MemRefValue):
            # use data pointer as key when data is a memref,
            # usually come from users, no need to deduplicate
            data = (op.data.ptr,)
        else:
            # small constants can be deduplicated
            # when data is a list
            data = list_to_tuple(op.data)

        # Create a unique key for the constant based on its data and type
        return (data, op.outputs[0].dtype, list_to_tuple(op.outputs[0].shape))

    def integrate_subgraph(self, inputs: List["FlatIRTensor"], outputs: List["FlatIRTensor"]) -> None:
        """
        Integrate a subgraph delineated by the given inputs and outputs into this FlatIR.
        """
        from nvtripy.flat_ir.function import FlatIRFunction
        from nvtripy.flat_ir.ops import ConstantOp
        from nvtripy.flat_ir.ops.base import BaseFlatIROp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        seen_tensors: Set[int] = set()
        dedup_func_op_map: Dict[int, List[FlatIRFunction]] = {}

        def _register_tensor_and_collect_ops(
            tensor: FlatIRTensor,
            seen_tensors: Set[int],
            new_ops: List[Union[BaseFlatIROp, FlatIRFunction]],
            tensor_replacements: Dict[str, FlatIRTensor],
            constant_map: Dict[str, FlatIRTensor],
        ) -> None:
            """
            Register a tensor and collect its producers using depth-first search.
            """
            if id(tensor) in seen_tensors:
                return

            seen_tensors.add(id(tensor))
            self.register_tensor(tensor)
            op = tensor.producer
            assert (
                op
            ), f"Tensor: {tensor} has no producer set. Did you use the constructor instead of the `build()` function?"

            if isinstance(op, FlatIRFunction):
                _process_function_op(op, new_ops, seen_tensors, inputs)
            else:
                _process_regular_op(op, new_ops, tensor, seen_tensors, inputs, tensor_replacements, constant_map)

        def _process_function_op(
            op: FlatIRFunction,
            new_ops: List[Union[BaseFlatIROp, FlatIRFunction]],
            seen_tensors: Set[int],
            inputs: List[FlatIRTensor],
        ) -> None:
            """
            Process a FlatIRFunction operation.
            """
            dedup_op = _find_duplicate_func_op(op)

            # Return early if an op or its dedup op is seen already.
            if not _is_first_visit(op, new_ops) or not _is_first_visit(dedup_op, new_ops):
                return

            # If a dedup op is seen for the first time, append to new ops and stop backward pass.
            if dedup_op:
                new_ops.append(dedup_op)
                return

            func_ops: List[BaseFlatIROp] = []
            func_tensor_replacements: Dict[str, FlatIRTensor] = {}
            func_constant_map: Dict[str, FlatIRTensor] = {}

            for inp in op.inputs:
                seen_tensors.add(id(inp))
                self.register_tensor(inp)

            for start_tensor in op.outputs:
                if start_tensor not in inputs:
                    _register_tensor_and_collect_ops(
                        start_tensor, seen_tensors, func_ops, func_tensor_replacements, func_constant_map
                    )

            _apply_tensor_replacements(func_ops, func_tensor_replacements)
            op.ops = func_ops
            new_ops.append(op)

        def _process_regular_op(
            op: BaseFlatIROp,
            new_ops: List[Union[BaseFlatIROp, FlatIRFunction]],
            tensor: FlatIRTensor,
            seen_tensors: Set[int],
            inputs: List[FlatIRTensor],
            tensor_replacements: Dict[str, FlatIRTensor],
            constant_map: Dict[str, FlatIRTensor],
        ) -> None:
            """
            Process a regular (non-function) operation.
            """
            # If a constant is already been declared in the flatIR, reuse that constant instead of redefining the constant.
            if isinstance(op, ConstantOp):
                constant_key = self._get_constant_key(op)
                if constant_key in constant_map:
                    # Reuse existing constant
                    existing_tensor = constant_map[constant_key]
                    tensor_replacements[tensor.name] = existing_tensor
                else:
                    # New unique constant, add to map
                    constant_map[constant_key] = op.outputs[0]
                    new_ops.append(op)
            else:
                for input_tensor in op.inputs:
                    if input_tensor not in inputs:
                        _register_tensor_and_collect_ops(
                            input_tensor, seen_tensors, new_ops, tensor_replacements, constant_map
                        )
                # Update op list with a "new" and "unique" op.
                new_ops.append(op)

        def _find_duplicate_func_op(op: FlatIRFunction) -> Union[FlatIRFunction, None]:
            """
            Find a structurally equivalent FlatIRFunction in the existing ops.
            """
            for existing_op in self.ops:
                if isinstance(existing_op, FlatIRFunction) and existing_op.is_structurally_equivalent(op):
                    dedup_func_op_map.setdefault(id(existing_op), []).append(op)
                    return existing_op
            return None

        def _is_first_visit(op: FlatIRFunction, new_ops: List[FlatIRFunction]) -> bool:
            """
            Check if this is the first visit to the operation.
            """
            return op is None or not any(id(op) == id(existing_op) for existing_op in new_ops)

        def _apply_tensor_replacements(
            ops: List[Union[BaseFlatIROp, FlatIRFunction]],
            replacements: Dict[str, FlatIRTensor],
        ) -> None:
            """
            Apply tensor replacements to the given operations.
            """
            for op in ops:
                if isinstance(op, FlatIRFunction):
                    op.set_caller_inputs(
                        [self.tensor_replacements.get(inp.name, inp) for inp in op.get_caller_inputs()]
                    )
                    # Perform replacements for all operation deduped by current op
                    if id(op) in dedup_func_op_map:
                        ops = dedup_func_op_map[id(op)]
                        for dop in ops:
                            dop.set_caller_inputs([replacements.get(inp.name, inp) for inp in dop.get_caller_inputs()])
                else:
                    op.inputs = [replacements.get(inp.name, inp) for inp in op.inputs]

        def _rebind_producers(
            inputs,
            outputs,
            ops: List[Union[BaseFlatIROp, FlatIRFunction]],
            dedup_func_op_map: Dict[int, List[FlatIRFunction]],
        ) -> None:
            """
            Rebind the producers to tensors from this FlatIR.
            """
            for op in ops:
                if isinstance(op, FlatIRFunction):
                    # If an existing op is replacing other ops, we need to rebind its mapping.
                    func_op = op

                    if id(op) in dedup_func_op_map:
                        # Dedup ops are inserted in backward iteration order. Pop from the left.
                        func_op = list(reversed(dedup_func_op_map[id(op)])).pop()

                        # Create a mapping from a common callee op to original inputs and outputs
                        caller_map = {
                            "caller_inputs": func_op.get_caller_inputs(),
                            "caller_outputs": func_op.get_caller_outputs(),
                        }
                        if op.caller_replacements:
                            op.caller_replacements.append(caller_map)
                        else:
                            op.caller_replacements = [caller_map]

                    for tensor in func_op.get_caller_inputs() + func_op.get_caller_outputs():
                        self.register_tensor(tensor)

                    # Since the common callee op now represent several ops, ensure we trace all such tensors.
                    op.trace_input_names = (op.trace_input_names or []) + [input_tensor.name for input_tensor in inputs]
                    op.trace_output_names = (op.trace_output_names or []) + [
                        output_tensor.name for output_tensor in outputs
                    ]
                else:
                    op.inputs = [self.register_tensor(input_tensor) for input_tensor in op.inputs]
                    op.outputs = [self.register_tensor(output_tensor) for output_tensor in op.outputs]
                    op.trace_input_names = [input_tensor.name for input_tensor in inputs]
                    op.trace_output_names = [output_tensor.name for output_tensor in outputs]

        def _process_main_ops(
            ops: List[Union[BaseFlatIROp, FlatIRFunction]]
        ) -> List[Union[BaseFlatIROp, FlatIRFunction]]:
            """
            Process the main operations, handling function replacements.
            """
            processed_ops = []
            for op in ops:
                if isinstance(op, FlatIRFunction):
                    if op.caller_replacements:
                        processed_ops.append(_process_function_replacements(op))
                    else:
                        processed_ops.append(op)
                        self.add_function(op)
                else:
                    processed_ops.append(op)
            return processed_ops

        def _process_function_replacements(op: FlatIRFunction) -> FlatIRFunction:
            """
            Process replacements for a FlatIRFunction.
            """
            caller_map = op.caller_replacements.pop()
            op_inputs = [
                input_tensor.clone(reason_details=f"Function input cloned from {input_tensor}", name=input_tensor.name)
                for input_tensor in op.inputs
            ]
            op_outputs = [
                output_tensor.clone(
                    reason_details=f"Function output cloned from {output_tensor}", name=output_tensor.name
                )
                for output_tensor in op.outputs
            ]
            for callee_input, caller_input in zip(op_inputs, caller_map["caller_inputs"]):
                callee_input.caller_tensor = caller_input
            for callee_output, caller_output in zip(op_outputs, caller_map["caller_outputs"]):
                callee_output.caller_tensor = caller_output
            return op.clone_with_new_io(op_inputs, op_outputs)

        # Main execution
        main_ops: List[Union[BaseFlatIROp, FlatIRFunction]] = []
        for start_tensor in outputs:
            _register_tensor_and_collect_ops(
                start_tensor, seen_tensors, main_ops, self.tensor_replacements, self.constant_map
            )

        _apply_tensor_replacements(main_ops, self.tensor_replacements)
        _rebind_producers(inputs, outputs, main_ops, dedup_func_op_map)
        self.ops.extend(_process_main_ops(main_ops))
