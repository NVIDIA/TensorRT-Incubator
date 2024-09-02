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
from tripy import utils
from tripy.common.shape_bounds import ShapeBounds

from tripy.flat_ir.ops import ConstantOp


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

    def __str__(self) -> str:
        """Generate a string representation of the FlatIR."""
        from tripy.flat_ir.ops.base import FlatIRFunction, BaseFlatIROp

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
                output_names = ", ".join(output.name for output in op.get_caller_outputs())
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

        Args:
            function (FlatIRFunction): The function to be added to the FlatIR.
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
        from tripy.backend.mlir.utils import make_ir_context, make_tensor_location
        from tripy.flat_ir.ops.base import FlatIRFunction, BaseFlatIROp

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

        def _fix_up_function_inputs(func, inputs, mlir_tensor_map):
            callee_types = []
            caller_types = []
            for calle_inp, caller_inp in zip(func.inputs, func.get_caller_inputs()):
                callee_types.append(mlir_tensor_map[calle_inp.name])
                caller_types.append(mlir_tensor_map[caller_inp.name])

            for input in inputs:
                if input in callee_types:
                    # Use calle index to index into caller function types.
                    calle_index = callee_types.index(input)
                    type = get_op_result_or_value(input).type
                    if get_op_result_or_value(caller_types[calle_index]).type != type:
                        # Update func output type for caller.
                        caller_input = get_op_result_or_value(caller_types[calle_index])
                        caller_input.set_type(type)

        def _func_op_to_mlir(insertion_point, func, mlir_tensor_map, func_map):
            with make_tensor_location(
                [inp.name for inp in func.inputs],
                [out.name for out in func.outputs],
                func.trace_input_names,
                func.trace_output_names,
            ):
                func_input_types = [
                    get_op_result_or_value(mlir_tensor_map[input_tensor.name]).type
                    for input_tensor in func.get_caller_inputs()
                ]
                # Use unknown types as function outputs are not yet materialized.
                func_output_types = [ir.UnrankedTensorType.get(ir.F32Type.get()) for _ in func.get_caller_outputs()]
                func_type = ir.FunctionType.get(func_input_types, func_output_types)
                func_op = func_dialect.FuncOp(func.name, func_type, ip=insertion_point, visibility="private")
                with ir.InsertionPoint(func_op.add_entry_block()):
                    # Map function inputs to MLIR tensor map
                    for index, input_tensor in enumerate(func.inputs):
                        mlir_tensor_map[input_tensor.name] = func_op.arguments[index]

                    # Process each operation in the function
                    for op in func.ops:
                        op_inputs = [mlir_tensor_map[input_tensor.name] for input_tensor in op.inputs]

                        op_outputs = op.to_mlir(op_inputs)

                        # It is possible that `op.to_mlir` could update the operation input types.
                        # e.g. It could resolve shape tensors types to be static.
                        # Ensure we fix up caller input types.
                        _fix_up_function_inputs(func, op_inputs, mlir_tensor_map)

                        # Map function inputs to MLIR tensor map
                        for mlir_output, flat_ir_output in zip(op_outputs, op.outputs):
                            mlir_tensor_map[flat_ir_output.name] = mlir_output

                    # Add return operation for the function
                    func_dialect.ReturnOp([mlir_tensor_map[output_tensor.name] for output_tensor in func.outputs])

                func_new_input_types = [get_op_result_or_value(mlir_tensor_map[o.name]).type for o in func.inputs]
                func_new_output_types = [get_op_result_or_value(mlir_tensor_map[o.name]).type for o in func.outputs]
                func_type = ir.FunctionType.get(func_new_input_types, func_new_output_types)
                func_op.attributes["function_type"] = ir.TypeAttr.get(func_type)

                func_inputs = [mlir_tensor_map[input_tensor.name] for input_tensor in func.get_caller_inputs()]
                func_call_op = func_dialect.CallOp(
                    func_new_output_types, ir.FlatSymbolRefAttr.get(func.name), func_inputs
                )
                # Create a map from function name to its call op and func op.
                func_map[func.name] = [func_call_op, func_op]

                for index, output_tensor in enumerate(func.get_caller_outputs()):
                    mlir_tensor_map[output_tensor.name] = func_call_op.results[index]

        def _fix_up_function_results(ops, mlir_tensor_map, func_map):
            # Iterate over all func ops and fix up its results based on func call op results types
            for func in ops:
                if not isinstance(func, FlatIRFunction):
                    continue
                call_op, func_op = func_map[func.name]
                assert len(call_op.results) == len(func.outputs)
                for call_op_res, func_output in zip(call_op.results, func.outputs):
                    # If there is mismatch between call op result type and corresponding caller result type,
                    # fix the call op result type and corresponding function type. `caller result type`
                    # could an input another function which could have its input type updated e.g. in `op.to_mlir`.
                    func_res = get_op_result_or_value(mlir_tensor_map[func_output.name])
                    if call_op_res.type != func_res.type:
                        # Update func result type.
                        func_res.set_type(call_op_res.type)
                        # Update function signature type.
                        res_index = func.outputs.index(func_output)
                        func_output_types = func_op.type.results
                        func_output_types[res_index] = func_res.type
                        new_func_type = ir.FunctionType.get(func_op.type.inputs, func_output_types)
                        func_op.attributes["function_type"] = ir.TypeAttr.get(new_func_type)

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
                    if bound.is_static():
                        arg_attrs.append(ir.DictAttr.get({}))
                    else:
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

                    func_map: Dict[str, [func_dialect.callOp, func_dialect.FuncOp]] = {}

                    with ir.InsertionPoint(main_func_op.add_entry_block()):
                        # Map main function inputs to MLIR tensor map
                        for index, input_tensor in enumerate(self.inputs):
                            mlir_tensor_map[input_tensor.name] = main_func_op.arguments[index]

                        # Process each op in the main flow
                        for op in self.ops:
                            if isinstance(op, FlatIRFunction):
                                _func_op_to_mlir(insertion_point, op, mlir_tensor_map, func_map)
                            elif isinstance(op, BaseFlatIROp):
                                _base_op_to_mlir(op, mlir_tensor_map)

                        # Add return operation for the main function
                        func_dialect.ReturnOp([mlir_tensor_map[output_tensor.name] for output_tensor in self.outputs])

                    # Fix up function results based on func caller op results types.
                    # Caller op result could have been updated while calling `op.to_mlir`.
                    # Ensure that we fix up the function signature which produced the result.
                    _fix_up_function_results(self.ops, mlir_tensor_map, func_map)

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

        from tripy.backend.mlir.utils import redirect_stderr

        try:
            with redirect_stderr() as outfile:
                mlir = to_mlir_impl()
        except Exception as exc:
            from tripy.backend.mlir.utils import map_error_to_user_code_and_raise

            outfile.flush()
            outfile.seek(0)
            stderr = outfile.read()

            map_error_to_user_code_and_raise(self, exc, stderr.decode())

        return mlir

    def register_tensor(self, tensor: "FlatIRTensor") -> "FlatIRTensor":
        """
        Registers a tensor with this FlatIR instance. If the tensor has no name, a name unique to this FlatIR will be assigned.

        Args:
            tensor: The tensor to register.
        """
        tensor.name = utils.default(tensor.name, f"t_inter{len(self.tensor_map)}")
        if tensor.name in self.tensor_map:
            return self.tensor_map[tensor.name]
        self.tensor_map[tensor.name] = tensor
        return tensor

    def _get_constant_key(self, op):
        from mlir_tensorrt.runtime._mlir_libs._api import MemRefValue
        from tripy.utils.utils import list_to_tuple, volume

        if isinstance(op.data, MemRefValue):
            from tripy.backend.mlir.memref import tolist

            VOLUME_THRESHOLD_FOR_MEMREF = 50
            if volume(op.data.shape) < VOLUME_THRESHOLD_FOR_MEMREF:
                l = tolist(op.data)
            else:
                l = [op.data.ptr]
            data = list_to_tuple(l if isinstance(l, List) else [l])
        elif isinstance(op.data, int) or isinstance(op.data, float) or isinstance(op.data, bool):
            data = list_to_tuple(
                op.data,
            )
        else:
            data = list_to_tuple(op.data)

        # Create a unique key for the constant based on its data and type
        return (data, op.outputs[0].dtype, list_to_tuple(op.outputs[0].shape))

    def integrate_subgraph(self, inputs: List["FlatIRTensor"], outputs: List["FlatIRTensor"]) -> None:
        """
        Integrate a subgraph delineated by the given inputs and outputs into this FlatIR.

        Args:
            inputs (List[FlatIRTensor]): The input tensors for the subgraph.
            outputs (List[FlatIRTensor]): The output tensors for the subgraph.
        """
        from tripy.flat_ir.ops.base import BaseFlatIROp, FlatIRFunction

        seen_tensors: Set[int] = set()

        # Ensure no constant duplication within a function.
        tensor_replacements: Dict[int, "FlatIRTensor"] = {}
        constant_map = {}

        def register_tensor_and_collect_ops(
            tensor: "FlatIRTensor",
            seen_tensors: Set[int],
            new_ops: List[Union[BaseFlatIROp, FlatIRFunction]],
            tensor_replacements,
            constant_map,
        ) -> None:
            """
            Register a tensor and collect its producers using depth-first search.

            Args:
                tensor (FlatIRTensor): The tensor to register.
                seen_tensors (Set[int]): A set of seen tensor IDs to avoid cycles.
            """

            if id(tensor) not in seen_tensors:
                seen_tensors.add(id(tensor))
                self.register_tensor(tensor)

                op = tensor.producer

                assert (
                    op is not None
                ), f"Tensor: {tensor} has no producer set. Did you use the constructor instead of the `build()` function?"

                if isinstance(op, FlatIRFunction) and not any(id(op) == id(existing_op) for existing_op in new_ops):
                    # Store nested function ops.
                    func_ops: List[BaseFlatIROp] = []

                    # Ensure no constant duplication within a function.
                    tensor_replacements: Dict[int, "FlatIRTensor"] = {}
                    constant_map = {}

                    for inp in op.inputs:
                        seen_tensors.add(id(inp))
                        self.register_tensor(inp)

                    # Register all tensors and collect nested ops from function outputs.
                    for start_tensor in op.outputs:
                        if start_tensor not in inputs:
                            register_tensor_and_collect_ops(
                                start_tensor, seen_tensors, func_ops, tensor_replacements, constant_map
                            )

                    # Apply tensor replacements in a nested function.
                    for nested_op in func_ops:
                        nested_op.inputs = [tensor_replacements.get(inp.name, inp) for inp in nested_op.inputs]

                    # Update ops to only relevant function ops i.e. ops reachable from the outputs.
                    op.ops = func_ops

                    # Update op list with a nested function op.
                    new_ops.append(op)
                else:
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
                                register_tensor_and_collect_ops(
                                    input_tensor, seen_tensors, new_ops, tensor_replacements, constant_map
                                )
                        # Update op list with a "new" and "unique" op.
                        new_ops.append(op)

        # Main op set
        main_ops: List[Union[BaseFlatIROp, FlatIRFunction]] = []
        for start_tensor in outputs:
            register_tensor_and_collect_ops(start_tensor, seen_tensors, main_ops, tensor_replacements, constant_map)

        # Apply tensor replacements in main function.new_ops
        for op in main_ops:
            op.inputs = [tensor_replacements.get(inp.name, inp) for inp in op.inputs]

        # Rebind the producers to tensors from this FlatIR
        for op in main_ops:
            if isinstance(op, FlatIRFunction):
                # Rebind the caller tensors to the callee tensors for a function.
                def rebind(caller, callee):
                    inputs = []
                    for caller_tensor, calle_tensor in zip(caller, callee):
                        inputs.append(self.register_tensor(caller_tensor))
                        setattr(calle_tensor, "caller_tensor", inputs[-1])
                    return inputs

                inputs = rebind(op.get_caller_inputs(), op.inputs)
                outputs = rebind(op.get_caller_outputs(), op.outputs)
                op.trace_input_names = [input_tensor.name for input_tensor in inputs]
                op.trace_output_names = [output_tensor.name for output_tensor in outputs]
            else:
                op.inputs = [self.register_tensor(input_tensor) for input_tensor in op.inputs]
                op.outputs = [self.register_tensor(output_tensor) for output_tensor in op.outputs]
                op.trace_input_names = [input_tensor.name for input_tensor in inputs]
                op.trace_output_names = [output_tensor.name for output_tensor in outputs]

        self.ops.extend(main_ops)
