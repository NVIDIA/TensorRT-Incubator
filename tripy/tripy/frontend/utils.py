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

import functools
import inspect
import numbers
from collections import deque
from typing import List, Sequence, Set, Union

from tripy import utils
from tripy.common.exception import raise_error
from tripy.flat_ir.function import FlatIRFunction
from tripy.flat_ir.ops import BaseFlatIROp


# Try to include correct column offsets for non-tensor arguments.
def _add_column_info_for_non_tensor(
    arg,
    arg_index,
    is_kwarg,
    dtype,
    num_args,
    func_name,
    skip_num_stack_entries,
    list_index=None,
    TensorType=None,
):
    from tripy.common.datatype import bool as tp_bool
    from tripy.common.datatype import floating, integer
    from tripy.frontend.shape import Shape
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.trace.ops.cast import cast

    TensorType = utils.default(TensorType, Tensor)

    assert not isinstance(
        arg, Tensor
    ), f"This function should not be called for objects that are already Tensor instances"

    if TensorType == Shape and isinstance(arg, numbers.Number):
        # Shapes require 1D values.
        arg = [arg]

    arg = TensorType(arg)
    arg.stack_info.fetch_source_code()

    # This is the stack depth in arg.stack_info where we find the decorated function.
    for idx, source_info in enumerate(arg.stack_info):
        if source_info.function == "wrapper" and source_info.module == __name__:
            WRAPPER_STACK_DEPTH = idx + 1
            break
    else:
        assert (
            False
        ), "`wrapper` function was not found in the call stack. Please update the check above if the name of the wrapper function has changed."

    # Find the first caller of this function that is NOT the function registry.
    # Also save the last dispatch target we see.
    dispatch_target = None
    for idx, source_info in enumerate(arg.stack_info[WRAPPER_STACK_DEPTH + skip_num_stack_entries :]):
        dispatch_target = source_info._dispatch_target or dispatch_target
        if source_info.module not in utils.get_module_names_to_exclude_from_stack_info():
            frame_index = idx + WRAPPER_STACK_DEPTH + skip_num_stack_entries
            break
    else:
        # Fallback path is just to look at the user code
        frame_index = arg.stack_info.get_first_user_frame_index()
        if frame_index is not None:
            dispatch_target = arg.stack_info[frame_index - 1]._dispatch_target

    source_info = arg.stack_info[frame_index]

    # The reverse binary ops need special handling since they will be swapped out for the non-reverse
    # variants and the order of operands will be inverted.
    REVERSE_BIN_OPS = {
        "__radd__",
        "__rmul__",
        "__rsub__",
        "__rpow__",
        "__rtruediv__",
    }

    if dispatch_target in REVERSE_BIN_OPS:
        assert arg_index in [0, 1]
        arg_index = 0 if arg_index == 1 else 1
        dispatch_target = dispatch_target.replace("__r", "__")

    # Special case for __getitem__: It is variadic. Argument 0 is the tensor,
    # and all subsequent arguments are slice parameters (in start, stop, step order).
    # Hence, we subtract one to get the index of the slice parameters
    if dispatch_target == "__getitem__":
        arg_index -= 1

    candidates = utils.get_arg_candidate_column_offsets(
        source_info.code,
        arg_index,
        num_args,
        dispatch_target if dispatch_target else func_name,
        is_kwarg,
        list_index=list_index,
    )

    # Only set column range if there is exactly one candidate, otherwise we can't reliably determine
    # what we should be pointing at.
    if len(candidates) == 1:
        source_info.column_range = candidates[0]

    if dtype is not None:
        # Refuse to do unsafe casts like float -> int.
        # NOTE: We do this check all the way down here so we have better stack information for `arg`.
        UNSAFE_CASTS = {floating: integer, integer: tp_bool}
        if any(issubclass(arg.dtype, frm) and issubclass(dtype, to) for frm, to in UNSAFE_CASTS.items()):
            raise_error(
                f"Refusing to automatically cast '{arg.dtype}' argument to '{dtype}'.",
                [
                    f"Hint: You may want to manually cast other operands in this expression to compatible types.\n",
                    "Note: argument was: ",
                    arg,
                ],
            )

        original_stack_info = arg.stack_info
        # TODO (pranavm): Remove `cast` and use dtype argument in Tensor constructor once `Shape`s are removed.
        arg = cast(arg, dtype=dtype)
        arg.stack_info = original_stack_info

    return arg


# NOTE: Conversion to tensors needs to be done via a decorator so that we can add stack information
# for non-tensors. Without having full context of the function signature, it is otherwise difficult to do so.
def convert_to_tensors(targets: Set[str] = None, skip_num_stack_entries: int = 0):
    """
    Decorator that converts specified arguments to tensors.
    If data type constraints are specified, any non-tensors converted will
    also be casted to the expected data type, but will raise an exception for lossy
    casts like float->int (but *not* for e.g. float32->float16).

    Args:
        targets: Names of arguments to convert to tensors. If not supplied, any arguments annotated
            with `TensorLike` are converted.

        skip_num_stack_entries: If the decorator is used on a function that is *called by*
            a function that the user invokes, it will be necessary to skip stack entries
            in order to get the column info from the user code. The number of entries skipped
            should be equal to the nesting depth from a function called by user code
            (if the decorated function is called by the user the depth is 0;
            if the decorated function is called from a user function, the depth is 1; etc.)
    """

    def impl(func):
        nonlocal targets

        from tripy.constraints import TYPE_VERIFICATION
        from tripy.types import TensorLike

        signature = inspect.signature(func)
        targets = targets or {name for name, param in signature.parameters.items() if param.annotation is TensorLike}

        constraints = {}
        if func.__qualname__ in TYPE_VERIFICATION:
            constraints = TYPE_VERIFICATION[func.__qualname__].constraints

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from tripy.frontend.tensor import Tensor
            from tripy.constraints import get_arg_dtype

            all_args = utils.merge_function_arguments(func, *args, **kwargs)

            # TODO (pranavm): Remove all this logic once Shapes are removed:

            from tripy.frontend.shape import Shape, ShapeScalar

            # Disallow mixing Tensor and Shape by default. If it makes sense in a given function
            # to have both Tensor and Shape arguments, that might suggest that custom handling
            # rather than relying on this decorator would make sense.
            types = {
                # There are other subclasses of Tensor, like Parameter and DefaultParameter.
                # Unless otherwise specified, we treat them as ordinary Tensors.
                Tensor if type(arg) not in {Shape, ShapeScalar} else type(arg)
                for _, arg in all_args
                if isinstance(arg, Tensor)
            }
            # We usually can treat ShapeScalars as either tensors or shapes due to broadcasting, so we can remove them from the below check.
            shape_scalar_encountered = ShapeScalar in types
            types -= {ShapeScalar}

            TensorType = None
            if types:
                TensorType = types.pop()
            # Result is a shape scalar only if we can't broadcast it up to anything else
            elif shape_scalar_encountered:
                TensorType = ShapeScalar

            # Materialize type variables from tensors.
            type_vars = {}
            for name, arg in all_args:
                if name in constraints:
                    dtype_result = get_arg_dtype(arg, func.__qualname__, name)
                    if dtype_result:
                        type_vars[constraints[name]] = dtype_result.value

            new_args = []
            new_kwargs = {}
            for index, (name, arg) in enumerate(all_args):

                def add_arg(arg):
                    if name not in kwargs:
                        new_args.append(arg)
                    else:
                        new_kwargs[name] = arg

                if name in targets and not isinstance(arg, Tensor):
                    dtype = None
                    if name in constraints and constraints[name] in type_vars:
                        dtype = type_vars[constraints[name]]

                    arg = _add_column_info_for_non_tensor(
                        arg,
                        index,
                        name in kwargs,
                        dtype,
                        len(args),
                        func.__name__,
                        skip_num_stack_entries,
                        TensorType=TensorType,
                    )

                add_arg(arg)

            return func(*new_args, **new_kwargs)

        return wrapper

    return impl


def _convert_nontensor_arg(
    arg,
    arg_idx,
    is_kwarg,
    num_args,
    func_name,
    skip_num_stack_entries,
    cast_dtype=None,
    list_index=None,
    TensorType=None,
):
    from tripy.utils import Result

    def is_valid_sequence(seq_arg: Sequence) -> Result:
        if len(seq_arg) == 0:
            return Result.ok()
        # If one is a sequence, all must be sequences of the same length. Do not accept strings.
        if isinstance(seq_arg[0], Sequence) and not isinstance(seq_arg[0], str):
            target_len = len(seq_arg[0])
            for inner_arg in seq_arg[1:]:
                if not isinstance(inner_arg, Sequence) or isinstance(inner_arg, str):
                    return Result.err([f"Expected a sequence but got {type(inner_arg).__qualname__}: {inner_arg}"])
                if len(inner_arg) != target_len:
                    return Result.err(
                        [f"Expected a sequence of length {target_len} but got length {len(inner_arg)}: {inner_arg}"]
                    )
                valid_inner = is_valid_sequence(inner_arg)
                if not valid_inner:
                    return valid_inner
            return Result.ok()
        # Otherwise check for numbers.
        for inner_arg in seq_arg:
            if not isinstance(inner_arg, numbers.Number):
                return Result.err(
                    [f"Encountered non-number of type {type(inner_arg).__qualname__} in sequence: {inner_arg}"]
                )
        return Result.ok()

    if isinstance(arg, Sequence):
        valid_sequence = is_valid_sequence(arg)
        if not valid_sequence:
            raise_error(f"Encountered invalid sequence argument: {arg}", details=valid_sequence.error_details)

    return _add_column_info_for_non_tensor(
        arg,
        arg_idx,
        is_kwarg=is_kwarg,
        dtype=cast_dtype,
        num_args=num_args,
        func_name=func_name,
        skip_num_stack_entries=skip_num_stack_entries,
        list_index=list_index,
        TensorType=TensorType,
    )


def convert_shape_inputs(targets: Sequence[str], skip_num_stack_entries: int = 0):
    """
    Decorator that converts the specified arguments to `Shape`s before passing them along
    to the decorated function. Converts only Tripy Tensors, Python numbers, or lists of Python numbers;
    other input formats like `numpy` arrays will not be accepted.

    If a targeted argument is a list, the individual members will be converted to `Shape`s
    and then concatenated together into a single `Shape`.

    Args:
        targets: List of args to the decorated function to convert.

        skip_num_stack_entries: If the decorator is used on a function that is *called by*
            a function that the user invokes, it will be necessary to skip stack entries
            in order to get the column info from the user code. The number of entries skipped
            should be equal to the nesting depth from a function called by user code
            (if the decorated function is called by the user the depth is 0;
            if the decorated function is called from a user function, the depth is 1; etc.)
    """

    def impl(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from tripy.common.exception import raise_error
            from tripy.frontend.shape import Shape
            from tripy.frontend.tensor import Tensor

            all_args = utils.merge_function_arguments(func, *args, **kwargs)

            new_args = []
            new_kwargs = {}
            for index, (name, arg) in enumerate(all_args):

                def add_arg(arg):
                    if name not in kwargs:
                        new_args.append(arg)
                    else:
                        new_kwargs[name] = arg

                def convert_nontensor_arg(arg, list_index=None):
                    if not isinstance(arg, numbers.Number) and not isinstance(arg, Sequence):
                        raise_error(
                            f"Unsupported input format {type(arg)} in argument {arg}. "
                            "Expected a Tripy Tensor or sequence of Python numbers."
                        )

                    return _convert_nontensor_arg(
                        arg,
                        index,
                        name in kwargs,
                        len(args),
                        func.__name__,
                        skip_num_stack_entries,
                        cast_dtype=None,
                        list_index=list_index,
                        TensorType=Shape,
                    )

                if name not in targets:
                    add_arg(arg)
                    continue

                from tripy.frontend.trace.ops.concatenate import concatenate
                from tripy.frontend.trace.ops.unsqueeze import unsqueeze

                if isinstance(arg, Shape):
                    add_arg(arg)
                    continue
                if isinstance(arg, Tensor):
                    add_arg(Shape(arg))
                    continue
                # if it's not a tensor and not a sequence, we treat as a singleton shape
                if not isinstance(arg, Sequence):
                    add_arg(convert_nontensor_arg([arg]))
                    continue

                # otherwise, for a sequence we convert all at once if no member is a tensor
                # and concatenate together with tensors if there is
                if len(arg) == 0:
                    add_arg(Shape([]))
                    continue

                if not any(map(lambda member: isinstance(member, Tensor), arg)):
                    add_arg(convert_nontensor_arg(arg))
                    continue

                shape_components = []
                # accumulate non-tensors together to be converted into shapes
                acc = []
                for member in arg:
                    if not isinstance(member, Tensor):
                        acc.append(member)
                        continue
                    if len(acc) > 0:
                        shape_components.append(convert_nontensor_arg(acc))
                        acc = []
                    if member.rank != 0:
                        raise_error("Tensor in a shape argument must be a scalar.", [f"Got {member}"])
                    member = Shape(unsqueeze(member, 0))
                    # Force the trace tensor shape to be (1,) since its known that we are reshaping a scalar to a 1D tensor.
                    # If we don't force the shape below, Tripy might require computing the shape of this trace tensor which can be expensive.
                    member.trace_tensor.shape = (1,)
                    shape_components.append(member)
                if len(acc) > 0:
                    shape_components.append(convert_nontensor_arg(acc))
                add_arg(concatenate(shape_components, 0))

            return func(*new_args, **new_kwargs)

        return wrapper

    return impl


def make_function(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.frontend.trace.ops.base import BaseTraceOp

        # Determine if this is a method or a free function.
        is_method = inspect.ismethod(func) or (inspect.isfunction(func) and args and isinstance(args[0], BaseTraceOp))

        # Generate a unique name for free function.
        fn_name = func.__qualname__ + "_" + str(id(func))

        if is_method:
            fn_name = args[0].__class__.__name__  # Use instance class name for methods i.e. `to_flat_ir`
            inputs = kwargs.get("inputs", args[1] if len(args) > 1 else [])
            outputs = kwargs.get("outputs", args[2] if len(args) > 2 else [])
        else:
            # Collect inputs from args and kwargs
            inputs = [arg for arg in args if isinstance(arg, FlatIRTensor)]
            inputs.extend([v for v in kwargs.values() if isinstance(v, FlatIRTensor)])
            outputs = []

        # Call the original function.
        result = func(*args, **kwargs)

        # For free functions, the result might be the output tensor.
        if not is_method:
            if isinstance(result, FlatIRTensor):
                outputs = [result]
            if isinstance(result, (list, tuple)) and all(isinstance(r, FlatIRTensor) for r in result):
                outputs = list(result)

        def collect_ops(inputs, outputs):
            ops = []
            visited_ops = set()
            input_ids = {id(input_tensor) for input_tensor in inputs}
            stack = [(output, None) for output in outputs]  # (tensor, parent_op)

            while stack:
                current, parent_op = stack.pop()

                if parent_op and id(parent_op) not in visited_ops:
                    ops.append(parent_op)
                    visited_ops.add(id(parent_op))

                if hasattr(current, "producer") and id(current) not in input_ids:
                    producer = current.producer
                    if id(producer) not in visited_ops:
                        visited_ops.add(id(producer))
                        ops.append(producer)
                        if producer.inputs:
                            stack.extend((input_tensor, producer) for input_tensor in producer.inputs)
                        else:
                            # Handle ops with no inputs
                            stack.append((current, producer))

            # Reverse ops to get them in the order they were originally executed.
            ops.reverse()

            return ops

        # Trace back from outputs to inputs.
        ops = collect_ops(inputs, outputs)

        # Create a mapping of original tensors to their clones.
        tensor_map = {
            id(tensor): tensor.clone(reason_details=f"clone tensor {tensor} for function input/output")
            for tensor in inputs + outputs
        }

        # Function to get or create a cloned tensor
        def get_or_create_cloned_tensor(tensor):
            if id(tensor) not in tensor_map:
                tensor_map[id(tensor)] = tensor.clone(reason_details=f"clone tensor {tensor} inside a function.")
            return tensor_map[id(tensor)]

        # Update ops with cloned tensors.
        for op in ops:
            op.inputs = [get_or_create_cloned_tensor(input_tensor) for input_tensor in op.inputs]
            op.outputs = [get_or_create_cloned_tensor(output_tensor) for output_tensor in op.outputs]
            # Set the producer for each output
            for output in op.outputs:
                output.producer = op

        # Update callee_inputs and callee_outputs
        callee_inputs = [tensor_map[id(input_tensor)] for input_tensor in inputs]
        callee_outputs = [tensor_map[id(output_tensor)] for output_tensor in outputs]

        # Create a map from callee tensor to caller tensors.
        for callee, caller in zip(callee_inputs + callee_outputs, inputs + outputs):
            setattr(callee, "caller_tensor", caller)

        # Finally create the flat ir function
        flat_ir_function = FlatIRFunction(fn_name, callee_inputs, callee_outputs, ops)

        # Set the producer of each output to be this FlatIRFunction.
        for output in outputs:
            output.producer = flat_ir_function

        # Return the original result if it is a free function.
        if not is_method:
            return result

    return wrapped


def topological_sort(ops: List[Union["BaseTraceOp", BaseFlatIROp]]) -> List[Union["BaseTraceOp", BaseFlatIROp]]:
    """
    This utility to topologically sort a graph that can be a Trace or a FlatIR graph.
    """
    stack = deque()
    visited_layer_ids = set()
    result_set = set()
    result = list()
    id_ops = set(id(op) for op in ops)

    for op in ops:
        if id(op) not in visited_layer_ids:
            stack.append((op, False))

            while stack:
                current_op, is_processed = stack.pop()
                if id(current_op) in result_set:
                    continue
                if is_processed:
                    result.append(current_op)
                    result_set.add(id(current_op))
                    continue

                visited_layer_ids.add(id(current_op))
                stack.append((current_op, True))

                for ip in reversed(current_op.inputs):
                    if (
                        ip.producer is not None
                        and id(ip.producer) not in visited_layer_ids
                        and id(ip.producer) in id_ops
                    ):
                        stack.append((ip.producer, False))

    assert len(ops) == len(result), f"Num original ops {len(ops)}, got num {len(result)}"
    return result


# Processes a `dim` (i.e. axis) argument related to a tensor.
# If the dimension is negative, this will convert it to the corresponding positive index.
#
# NOTE: This is currently an extremely specialized decorator that expects an `input` tensor
# argument and a `dim` argument for the axis.
# In the future, we can generalize this and use it more broadly.
def process_dim(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        input = utils.get_arg_by_name("input", func, *args, **kwargs)

        def process_dim(dim: int) -> int:
            new_dim = dim
            if dim < 0:
                new_dim = input.rank + dim

            if new_dim < 0 or new_dim >= input.rank:
                raise_error(
                    "Dimension argument is out of bounds.",
                    [
                        f"Note: provided dimension was: {dim}, while the tensor has a rank of: {input.rank}.\n"
                        f"Dimension should be in the half-open interval: [{-input.rank}, {input.rank})."
                    ],
                )
            return new_dim

        args, kwargs = utils.modify_arg("dim", process_dim, func, *args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def pretty_print(data_list, shape, threshold=1000, linewidth=10, edgeitems=3):
    """
    Returns a pretty-print string of list format data.
    """

    def _data_str(data, summarize, linewidth, edgeitems, indent=0):
        if isinstance(data, (float, int)):
            return str(data)

        if len(data) == 0 or isinstance(data[0], (float, int)):
            if summarize and len(data) > 2 * edgeitems:
                data_lines = [data[:edgeitems] + [" ..."] + data[-edgeitems:]]
            else:
                data_lines = [data[i : i + linewidth] for i in range(0, len(data), linewidth)]
            lines = [", ".join([f"{e:.4f}" if isinstance(e, float) else str(e) for e in line]) for line in data_lines]
            return "[" + ("," + "\n" + " " * (indent + 1)).join(lines) + "]"

        if summarize and len(data) > 2 * edgeitems:
            slices = (
                [_data_str(data[i], summarize, linewidth, edgeitems, indent + 1) for i in range(0, edgeitems)]
                + ["..."]
                + [
                    _data_str(data[i], summarize, linewidth, edgeitems, indent + 1)
                    for i in range(len(data) - edgeitems, len(data))
                ]
            )
        else:
            slices = [_data_str(data[i], summarize, linewidth, edgeitems, indent + 1) for i in range(0, len(data))]

        tensor_str = ("," + "\n" * (max(len(shape) - indent - 1, 1)) + " " * (indent + 1)).join(slices)
        return "[" + tensor_str + "]"

    numel = 1
    for d in shape:
        numel *= d
    summarize = numel > threshold
    return _data_str(data_list, summarize, linewidth, edgeitems)
