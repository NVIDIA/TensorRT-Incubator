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
import numbers
from collections import deque
from typing import List, Optional, Sequence, Tuple, Union

from tripy import utils
from tripy.common.exception import raise_error
from tripy.flat_ir.ops import BaseFlatIROp
from tripy.frontend.trace.ops import BaseTraceOp


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
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.shape import Shape
    from tripy.frontend.trace.ops.cast import cast

    TensorType = utils.default(TensorType, Tensor)

    assert not isinstance(
        arg, Tensor
    ), f"This function should not be called for objects that are already Tensor instances"

    if isinstance(arg, numbers.Number) and TensorType == Shape:
        # Shapes require 1D values.
        arg = [arg]

    arg = TensorType(arg)
    if dtype is not None:
        arg = cast(arg, dtype=dtype)
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
    # Hence, we subtract two to get the index of the slice parameters
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

    return arg


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


"""
Non-magic methods that are allowed to be decorated with convert_inputs_to_tensors.
We should generally avoid exceptions.
"""
CONVERT_TENSOR_EXCEPTIONS = {
    "slice_helper",  # We use it to convert inputs to __getitem__ but need to handle slices before converting
}


# Decorator to preprocess inputs of a function and convert Python numbers to tripy tensors.
def convert_inputs_to_tensors(
    sync_arg_types: Optional[List[Tuple[str]]] = None,
    exclude: Optional[List[str]] = None,
    unpack_argument: Optional[List[str]] = None,
    skip_num_stack_entries: int = 0,
):
    """
    Decorator that converts all arguments to `Tensor`s before passing them along
    to the decorated function. Converts only Python numbers or lists of Python numbers;
    inputs like `numpy` arrays will be left unchanged and should be handled manually.

    This decorator is intended mainly to be used with overloads of Python operators; for other
    cases, we recommend that users explicitly convert non-Tripy inputs.

    Args:
        sync_arg_types: A list of tuples of strings indicating the parameter indices for parameters
            that must share a type. For example, `sync_arg_types=[("a", "b"), ("c", "d")]` indicates that
            arguments `a` and `b` and arguments `c` and `d` should have the same types. Type casting is only
            enabled for Python numbers, and at least one of the arguments in each tuple must be a `Tensor`.
            For arguments that are lists included in `unpack_argument`,
            the syncing will be done for each member of the specified lists.

        exclude: A list of names of arguments to skip over. These arguments will not be modified.

        unpack_argument: If an argument name is included and it is a list,
          the members of the list will each be individually converted into `Tensor`s,
          rather than having the whole list converted into a `Tensor`.

        skip_num_stack_entries: If the decorator is used on a function that is *called by*
            a function that the user invokes, it will be necessary to skip stack entries
            in order to get the column info from the user code. The number of entries skipped
            should be equal to the nesting depth from a function called by user code
            (if the decorated function is called by the user the depth is 0;
            if the decorated function is called from a user function, the depth is 1; etc.)
    """

    sync_arg_types = utils.default(sync_arg_types, [])
    exclude = utils.default(exclude, [])
    unpack_argument = utils.default(unpack_argument, [])

    def impl(func):

        func_name = func.__name__
        # only checking __ at the start and end; in principle, we could add an exhaustive list of magic methods
        if (
            not func_name.startswith("__") or not func_name.endswith("__")
        ) and func_name not in CONVERT_TENSOR_EXCEPTIONS:
            raise_error(
                f"convert_inputs_to_tensors decorator is only permitted for magic methods. Decorated function: {func_name}"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from tripy.frontend.tensor import Tensor
            from tripy.frontend.shape import Shape, ShapeScalar

            all_args = utils.merge_function_arguments(func, *args, **kwargs)

            # TODO (#233): Disallow mixing Tensor/Shape. The workaround below works
            # because ShapeScalars will typically be broadcasted in order to operate
            # with Tensors/Shapes. Otherwise, Shape and ShapeScalar would have been
            # at the same level of hierarchy.
            #
            # When a function has multiple arguments, the order of precedence is:
            #
            # 1. Tensor
            # 2. Shape
            # 3. ShapeScalar
            #
            # That is, if we have an operation with a mixture of types, all arguments are
            # converted to the type with the highest precedence, e.g. Tensor + Shape -> Tensor.
            TensorType = None
            MaxType = max(
                (type(arg) for arg_name, arg in all_args if arg_name not in exclude),
                key=lambda typ: {Tensor: 2, Shape: 1, ShapeScalar: 0}.get(typ, -1),
            )
            if issubclass(MaxType, Tensor):
                TensorType = MaxType

            def get_arg(name: str):
                for arg_name, arg in all_args:
                    if name == arg_name:
                        return arg

                assert (
                    False
                ), f"Cannot retrieve unnamed argument. This could be because the argument ({name}) is a variadic argument."

            new_args = []
            new_kwargs = {}
            for index, (name, arg) in enumerate(all_args):

                def add_arg(arg):
                    if name not in kwargs:
                        new_args.append(arg)
                    else:
                        new_kwargs[name] = arg

                def find_sync_target_dtype(arg_name):

                    for sync_tuple in sync_arg_types:
                        if arg_name not in sync_tuple:
                            continue

                        for tensor_name in sync_tuple:
                            sync_target = get_arg(tensor_name)
                            # If multiple Tensors exist in a tuple,
                            # leave the dtype check to frontend ops

                            if isinstance(sync_target, Sequence):
                                for member in sync_target:
                                    if isinstance(member, Tensor):
                                        return member.dtype
                            elif isinstance(sync_target, Tensor):
                                return sync_target.dtype
                        else:
                            sync_args = {sync_arg_name: get_arg(sync_arg_name) for sync_arg_name in sync_tuple}
                            raise_error(
                                f"At least one of the arguments: {sync_tuple} must be a `tripy.Tensor`.",
                                [f"Got {sync_args}"],
                            )
                    return None

                def convert_nontensor_arg(arg, list_index=None):
                    # simply do not convert in these cases and let the registry give an error instead
                    if not isinstance(arg, numbers.Number) and not isinstance(arg, Sequence):
                        return arg

                    return _convert_nontensor_arg(
                        arg,
                        index,
                        name in kwargs,
                        len(args),
                        func.__name__,
                        skip_num_stack_entries,
                        find_sync_target_dtype(name),
                        list_index=list_index,
                        TensorType=TensorType,
                    )

                if name in exclude or isinstance(arg, Tensor):
                    add_arg(arg)
                    continue
                if name in unpack_argument and isinstance(arg, Sequence):
                    new_list = [
                        member if isinstance(member, Tensor) else convert_nontensor_arg(member, list_index=i)
                        for i, member in enumerate(arg)
                    ]
                    if isinstance(arg, tuple):
                        new_list = tuple(new_list)
                    add_arg(new_list)
                    continue
                add_arg(convert_nontensor_arg(arg))

            return func(*new_args, **new_kwargs)

        return wrapper

    return impl


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
                    shape_components.append(member)
                if len(acc) > 0:
                    shape_components.append(convert_nontensor_arg(acc))
                add_arg(concatenate(shape_components, 0))

            return func(*new_args, **new_kwargs)

        return wrapper

    return impl


def topological_sort(ops: List[Union[BaseTraceOp, BaseFlatIROp]]) -> List[Union[BaseTraceOp, BaseFlatIROp]]:
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
