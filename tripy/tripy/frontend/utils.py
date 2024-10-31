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
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from tripy import utils
from tripy.common.exception import raise_error
from tripy.flat_ir.function import FlatIRFunction
from tripy.flat_ir.ops import BaseFlatIROp
from tripy.types import ShapeLike, TensorLike


def tensor_from_shape_like(arg: ShapeLike) -> "tripy.Tensor":
    from tripy.common.datatype import int32
    from tripy.frontend.dimension_size import DimensionSize
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.trace.ops.concatenate import concatenate
    from tripy.frontend.trace.ops.reshape import Reshape

    if not arg:
        return Tensor([], dtype=int32)

    concat_tensors = []

    # We accumulate integers so we can create just a single tensor for each contiguous
    # sequence of integers.
    int_buffer = []

    def empty_buffer():
        if not int_buffer:
            return

        concat_tensors.append(Tensor(int_buffer, dtype=int32))
        int_buffer.clear()

    for elem in arg:
        if isinstance(elem, DimensionSize):
            empty_buffer()
            # NOTE: We cannot use the reshape API here since it would lead to an
            # infinite loop when attempting to convert the shape input to a tensor.
            concat_tensors.append(Reshape.build([elem, Tensor([1])], 1))
        else:
            int_buffer.append(elem)

    empty_buffer()

    out = concatenate(concat_tensors, dim=0)
    # We must set the shape of the shape tensor here since otherwise we will not be able
    # to infer ranks in the frontend. Note that the reshape operations above will not result
    # in a tensor with known shapes even though the new shape is actually known.
    out.trace_tensor.shape = [len(arg)]
    return out


# Try to include correct column offsets for non-tensor arguments.
def _add_column_info(arg, arg_index, is_kwarg, num_positional, func_name, skip_num_stack_entries, arg_names):
    from tripy.frontend.tensor import Tensor

    assert isinstance(arg, Tensor), f"This function should only be called for objects that are already Tensor instances"

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
        source_info.code, arg_index, num_positional, dispatch_target or func_name, is_kwarg, arg_names
    )

    # Only set column range if there is exactly one candidate, otherwise we can't reliably determine
    # what we should be pointing at.
    if len(candidates) == 1:
        source_info.column_range = candidates[0]


# NOTE: Conversion to tensors needs to be done via a decorator so that we can add stack information
# for non-tensors. Without having full context of the function signature, it is otherwise difficult to do so.
def convert_to_tensors(
    targets: Set[str] = None, skip_num_stack_entries: int = 0, preprocess_args: Optional[Callable] = None
):
    """
    Decorator that converts specified arguments to Tensors or DimensionSizes.
    If the argument can be converted to a DimensionSize, it is. Otherwise, it is
    converted to a Tensor.

    If data type constraints are specified, any non-tensors converted will
    also be casted to the expected data type, but will raise an exception for lossy
    casts like float->int (but *not* for e.g. float32->float16).

    Args:
        targets: Names of arguments to convert to tensors. If not supplied, any arguments annotated
            with `TensorLike` or `ShapeLike` are converted.

        skip_num_stack_entries: If the decorator is used on a function that is *called by*
            a function that the user invokes, it will be necessary to skip stack entries
            in order to get the column info from the user code. The number of entries skipped
            should be equal to the nesting depth from a function called by user code
            (if the decorated function is called by the user the depth is 0;
            if the decorated function is called from a user function, the depth is 1; etc.).

            NOTE: When using this, make sure any extra arguments to the decorated function are
            passed as keyword arguments. Otherwise, the logic for determining column information
            will break.

        preprocess_args: A callback used to preprocess arguments before potential conversion. If provided,
            this is always called, regardless of whether the decorator actually needed to perform conversion.
            This will be called with all arguments that were passed to the decorated function and should
            return a dictionary of all updated arguments. For a variadic arg, the dictionary entry for the name
            should have a list of all the updated values.
    """

    def impl(func):
        nonlocal targets

        from tripy.constraints import TYPE_VERIFICATION

        signature = inspect.signature(func)
        targets = targets or {
            name for name, param in signature.parameters.items() if param.annotation in {TensorLike, ShapeLike}
        }
        shape_likes = {name for name, param in signature.parameters.items() if param.annotation is ShapeLike}

        constraints = {}
        if func.__qualname__ in TYPE_VERIFICATION:
            constraints = TYPE_VERIFICATION[func.__qualname__].constraints

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from tripy.common.datatype import bool as tp_bool
            from tripy.common.datatype import floating, integer
            from tripy.constraints import get_arg_dtype
            from tripy.frontend.dimension_size import DimensionSize
            from tripy.frontend.tensor import Tensor
            from tripy.frontend.trace.ops.cast import cast

            all_args = utils.merge_function_arguments(func, *args, **kwargs)

            if preprocess_args is not None:

                # Since a Python function can have at most one variadic arg and merge_function_arguments
                # would list all the entries consecutively, it suffices to find the name and start index
                # for the variadic arg to handle variadic args when preprocessing.
                def find_variadic_name_and_start_idx(all_args: Sequence) -> Optional[Tuple[str, int]]:
                    encountered_names = {}
                    for i, (name, _) in enumerate(all_args):
                        if name in encountered_names:
                            return name, encountered_names[name]
                        if name not in encountered_names:
                            encountered_names[name] = i

                    return None

                var_arg_name, var_arg_start_idx = utils.default(
                    find_variadic_name_and_start_idx(all_args), (None, None)
                )
                new_args = preprocess_args(*args, **kwargs)
                for index in range(len(all_args)):
                    name, _ = all_args[index]
                    if name in new_args:
                        if name == var_arg_name:
                            assert var_arg_start_idx is not None
                            all_args[index] = (name, new_args[name][index - var_arg_start_idx])
                        else:
                            all_args[index] = (name, new_args[name])

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
                    if name in shape_likes:
                        arg = tensor_from_shape_like(arg)
                    else:
                        # Python integers can always be casted to the most restrictive type, which is DimensionSize in Tripy.
                        # DimensionSize can always be casted up to Tensor if needed, but the reverse is not true.
                        # NOTE: We do not use isinstance here because bool is a subclass of int.
                        arg = DimensionSize(arg) if type(arg) is int else Tensor(arg)

                    _add_column_info(
                        arg,
                        index,
                        name in kwargs,
                        len(args),
                        func.__name__,
                        skip_num_stack_entries,
                        [name for name, _ in all_args],
                    )

                    dtype = None
                    if name in constraints and constraints[name] in type_vars:
                        dtype = type_vars[constraints[name]]

                    if dtype is not None:
                        # Refuse to do unsafe casts like float -> int.
                        # NOTE: We do this check all the way down here so we have better stack information for `arg`.
                        UNSAFE_CASTS = {floating: integer, integer: tp_bool}
                        if any(
                            issubclass(arg.dtype, frm) and issubclass(dtype, to) for frm, to in UNSAFE_CASTS.items()
                        ):
                            raise_error(
                                f"Refusing to automatically cast '{arg.dtype}' argument to '{dtype}'.",
                                [
                                    f"Hint: You may want to manually cast other operands in this expression to compatible types.\n",
                                    "Note: argument was: ",
                                    arg,
                                ],
                            )

                        original_stack_info = arg.stack_info
                        arg = cast(arg, dtype=dtype)
                        arg.stack_info = original_stack_info

                add_arg(arg)

            return func(*new_args, **new_kwargs)

        return wrapper

    return impl


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
def process_dim(dim: int, input_rank: int) -> int:
    new_dim = dim
    if dim < 0:
        new_dim = input_rank + dim

    if new_dim < 0 or new_dim >= input_rank:
        raise_error(
            "Dimension argument is out of bounds.",
            [
                f"Note: provided dimension was: {dim}, while the tensor has a rank of: {input_rank}.\n"
                f"Dimension should be in the half-open interval: [{-input_rank}, {input_rank})."
            ],
        )
    return new_dim


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
