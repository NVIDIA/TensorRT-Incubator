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

import inspect
import functools
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Sequence

from tripy import config, utils
from tripy.common.exception import raise_error

TYPE_VERIFICATION = {}
RETURN_VALUE = "__RETURN_VALUE"


# Try to include correct column offsets for non-tensor arguments.
def _add_column_info(arg, arg_index, is_kwarg, num_positional, func_name, arg_names):
    from tripy.frontend.tensor import Tensor

    assert isinstance(arg, Tensor), f"This function should only be called for objects that are already Tensor instances"

    arg.stack_info.fetch_source_code()

    # This is the stack depth in arg.stack_info where we find the function where we are converting types.
    # Note: We cannot simply search for the wrapper, since the wrapper would appear on functions that stipulate type constraints
    # but do not convert types.
    for idx, source_info in enumerate(arg.stack_info):
        if source_info.function == convert_input_types.__name__ and source_info.module == __name__:
            WRAPPER_STACK_DEPTH = idx + 1
            break
    else:
        assert (
            False
        ), "`convert_input_types` function was not found in the call stack. Please update the check above if the name of the conversion function has changed."

    # Find the first caller of this function that is NOT the function registry.
    # Also save the last dispatch target we see.
    dispatch_target = None
    for idx, source_info in enumerate(arg.stack_info[WRAPPER_STACK_DEPTH:]):
        dispatch_target = source_info._dispatch_target or dispatch_target
        if source_info.module not in utils.get_module_names_to_exclude_from_stack_info():
            frame_index = idx + WRAPPER_STACK_DEPTH
            break
    else:
        # Fallback path is just to look at the user code
        frame_index = arg.stack_info.get_first_user_frame_index()
        if frame_index is not None:
            dispatch_target = arg.stack_info[frame_index - 1]._dispatch_target

    # The function registry might prepend a class name to the dispatch target. We will strip it out here in order to match it.
    if dispatch_target is not None and "." in dispatch_target:
        dispatch_target = dispatch_target.split(".")[-1]

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

    candidates = utils.get_arg_candidate_column_offsets(
        source_info.code, arg_index, num_positional, dispatch_target or func_name, is_kwarg, arg_names
    )

    # Only set column range if there is exactly one candidate, otherwise we can't reliably determine
    # what we should be pointing at.
    if len(candidates) == 1:
        source_info.column_range = candidates[0]


def get_arg_dtype(arg, func_name, arg_name) -> utils.Result["tripy.dtype"]:
    from tripy.common.datatype import dtype
    from tripy.frontend.tensor import Tensor

    if isinstance(arg, Sequence):
        arg_dtypes = []
        for elem in arg:
            dtype_result = get_arg_dtype(elem, func_name, arg_name)
            if not dtype_result:
                return utils.Result.err(
                    [f"Could not determine data type of elements in sequence: {arg_name}"] + dtype_result.error_details
                )
            arg_dtypes.append(dtype_result.value)

        if len(set(arg_dtypes)) != 1:
            return utils.Result.err(
                [
                    f"Mismatched data types in sequence argument for '{func_name}'.\n",
                    f"For parameter: '{arg_name}', all arguments must have the same data type, but got: "
                    f"{arg_dtypes}",
                ],
            )
        arg_dtype = arg_dtypes[0]
    elif isinstance(arg, Tensor):
        arg_dtype = arg.dtype
    elif isinstance(arg, dtype):
        arg_dtype = arg
    else:
        return utils.Result.err([f"Expected a tensor or data type argument for {arg_name}, but got: {arg}"])
    return utils.Result.ok(arg_dtype)


def convert_input_types(func, args, kwargs, conversion_targets, conversion_preprocess_func, constraints, shape_likes):
    from tripy.common.datatype import bool as tp_bool
    from tripy.common.datatype import floating, integer
    from tripy.frontend.dimension_size import DimensionSize
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.utils import tensor_from_shape_like

    from tripy.frontend.trace.ops.cast import cast

    all_args, var_arg_info = utils.merge_function_arguments(func, *args, **kwargs)

    if conversion_preprocess_func is not None:
        var_arg_name, var_arg_start_idx = utils.default(var_arg_info, (None, None))
        new_args = conversion_preprocess_func(*args, **kwargs)
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

        if name in conversion_targets and not isinstance(arg, Tensor):
            if name in shape_likes:
                arg = tensor_from_shape_like(arg)
            else:
                # Python integers can always be cast to the most restrictive type, which is DimensionSize in Tripy.
                # DimensionSize can always be cast up to Tensor if needed, but the reverse is not true.
                # NOTE: We do not use isinstance here because bool is a subclass of int.
                arg = DimensionSize.create_directly(arg) if type(arg) is int else Tensor.create_directly(arg)

            _add_column_info(
                arg,
                index,
                name in kwargs,
                len(args),
                func.__name__,
                [name for name, _ in all_args],
            )

            dtype = None
            if name in constraints and constraints[name] in type_vars:
                dtype = type_vars[constraints[name]]

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
                arg = cast(arg, dtype=dtype)
                arg.stack_info = original_stack_info

        add_arg(arg)

    return new_args, new_kwargs


def dtypes(
    constraints: Dict[str, str] = {},
    variables: Dict[str, List[str]] = {},
    exceptions: List[Dict[str, str]] = [],
    aliases: List[str] = [],
    convert_tensor_and_shape_likes: bool = False,
    conversion_targets: Set[str] = None,
    conversion_preprocess_func: Optional[Callable] = None,
):
    """
    Specifies data type constraints for the decorated function. If `convert_tensor_and_shape_likes` is
    set to true, the decorated function will also convert the specified arguments to Tensors or DimensionSizes.

    NOTE: When annotating a new API, you should also update `tests/constraints/object_builders.py`.

    Args:
        constraints: Maps parameters and return values to data type constraint variables.
            Use the special value `constraints.RETURN_VALUE` to denote return values.
            For example:
                {"input": "T1", "other": T2, constraints.RETURN_VALUE: "T1"}

        variables: Maps data type constraints variables to their supported data types.
            For example:
                {"T1": ["float32", "float16"], "T2": ["int32", "int64"]}

        exceptions: Indicates specific combinations of data types that are not supported by the API.
            For example:
                [
                    {"T1": "float16", "T2": "int32"},
                ]

        aliases: A list of function name aliases. For methods that are exposed as multiple APIs
            (e.g. `__add__` and `__radd__`), this will enable type information to be added to the
            documentation for the aliases as well.

        convert_tensor_and_shape_likes: If set to True, this decorator will also convert
            arguments to `Tensor` or `DimensionSize`. If the argument can be converted to a
            `DimensionSize`, it is. Otherwise, it is converted to a `Tensor`.

            The conversions will respect any datatype constraints, casting the TensorLike values as necessary,
            but will raise an exception for lossy casts like float to int (but *not* for, e.g., `float32` to `float16`).

        conversion_targets: If `convert_tensor_and_shape_likes` is true, only the arguments named in the list
            will be converted to `Tensor` or `DimensionSize`. If not supplied, any arguments annotated
            with `TensorLike` or `ShapeLike` are converted if convert_tensor_and_shape_likes is true.

        conversion_preprocess_func: If `convert_tensor_and_shape_likes` is true, this argument is a callback that is
            used to preprocess the arguments before potential conversion. In this case, if provided, the callback
            will be called regardless of whether the decorator performs any conversions.

            The callback will be called with all arguments that were passed to the decorated function and should
            return a dictionary of all updated arguments. For a variadic arg, the dictionary entry for the name
            should have a list of all the updated values.
    """

    def decorator(func):
        nonlocal conversion_targets

        from tripy.types import ShapeLike, TensorLike

        return_dtype = constraints.get(RETURN_VALUE, None)
        VerifInfo = namedtuple("VerifInfo", ["obj", "inputs", "exceptions", "return_dtype", "dtypes", "constraints"])
        verif_info = VerifInfo(func, {}, exceptions, return_dtype, variables, constraints)

        signature = inspect.signature(func)
        conversion_targets = conversion_targets or {
            name for name, param in signature.parameters.items() if param.annotation in {TensorLike, ShapeLike}
        }
        shape_likes = {name for name, param in signature.parameters.items() if param.annotation is ShapeLike}

        for key in [func.__qualname__] + aliases:
            TYPE_VERIFICATION[key] = verif_info

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if convert_tensor_and_shape_likes:
                args, kwargs = convert_input_types(
                    func, args, kwargs, conversion_targets, conversion_preprocess_func, constraints, shape_likes
                )

            if config.enable_dtype_checking:
                from tripy.common.datatype import dtype
                from tripy.frontend.tensor import Tensor

                merged_args, _ = utils.merge_function_arguments(func, *args, **kwargs)

                # The first arguments seen for each type variable. Other arguments with the same variable
                # must use the same data types.
                type_var_first_args: Dict[str, Tuple[str, dtype, Any]] = {}

                for name, arg in merged_args:
                    if name not in constraints:
                        continue

                    if arg is None:
                        # This is only possible for omitted optional arguments. Otherwise, None will
                        # be disallowed by the function registry's type checking.
                        continue

                    type_var = constraints[name]

                    arg_dtype = get_arg_dtype(arg, func.__qualname__, name)
                    if not arg_dtype:
                        raise_error(f"Could not determine datatype of {name}.", arg_dtype.error_details)
                    arg_dtype = arg_dtype.value

                    # Check if the type is supported at all
                    supported_dtypes = variables[type_var]
                    if arg_dtype.name not in supported_dtypes:
                        raise_error(
                            f"Unsupported data type for '{func.__qualname__}'.",
                            [
                                f"For parameter: '{name}', got unsupported data type: '{arg_dtype}'.\n"
                                f"Supported data types are: {supported_dtypes}.\n"
                            ]
                            + (
                                [
                                    f"Note: '{name}' was: ",
                                    arg,
                                ]
                                if isinstance(arg, Tensor)
                                else []
                            ),
                        )

                    # Check if the type matches that of other inputs with the same type_var.
                    if type_var in type_var_first_args:
                        other_name, other_arg_dtype, other_arg = type_var_first_args[type_var]
                        if other_arg_dtype != arg_dtype:
                            raise_error(
                                f"Mismatched data types for '{func.__qualname__}'.",
                                [
                                    f"Parameters: '{other_name}' and '{name}' must have matching data types, but got: "
                                    f"'{other_arg_dtype.name}' and '{arg_dtype.name}' respectively.\n"
                                ]
                                + (
                                    [
                                        f"Note: '{other_name}' was: ",
                                        other_arg,
                                        f"While '{name}' was: ",
                                        arg,
                                    ]
                                    if isinstance(arg, Tensor)
                                    else []
                                ),
                            )

                    type_var_first_args[type_var] = (name, arg_dtype, arg)

            return func(*args, **kwargs)

        return wrapper

    return decorator
