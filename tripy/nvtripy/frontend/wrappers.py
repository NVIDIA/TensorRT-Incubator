#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass
from textwrap import indent
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from nvtripy import config, utils
from nvtripy.common.datatype import dtype as tp_dtype
from nvtripy.common.exception import raise_error
from nvtripy.frontend.constraints import AlwaysTrue, Constraints, Equal, GetInput, GetDataType, Fetcher, doc_str
from nvtripy.frontend.constraints.optimizer import optimize_constraints


@dataclass
class OperatorConstraints:
    func: Callable
    input_requirements: Optional[Constraints]
    output_guarantees: Optional[Constraints]


# A list of tuples of (input_requirements, output_guarantees) for operators.
OPERATOR_CONSTRAINTS: List[OperatorConstraints] = []


# Try to include correct column offsets for non-tensor arguments.
def _add_column_info(arg, arg_index, is_kwarg, num_positional, func_name):
    from nvtripy.frontend.tensor import Tensor

    assert isinstance(arg, Tensor), f"This function should only be called for objects that are already Tensor instances"

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
        if source_info.module not in utils.stack_info.get_module_names_to_exclude_from_stack_info():
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
    source_info.fetch_source_code()

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

    candidates = utils.ast.get_arg_candidate_column_offsets(
        source_info.code, arg_index, num_positional, dispatch_target or func_name, is_kwarg
    )

    # Only set column range if there is exactly one candidate, otherwise we can't reliably determine
    # what we should be pointing at.
    if len(candidates) == 1:
        source_info.column_range = candidates[0]


def _find_known_datatypes(merged_args: List[Tuple[str, Any]], input_requirements: Constraints) -> Dict[str, tp_dtype]:
    """
    Identify known datatypes from input requirements to enable automatic type casting.

    This function searches for Equal constraints in the input requirements to determine
    which arguments should have matching datatypes. It skips Equal constraints that appear
    inside If statement conditions, as those represent conditional checks rather than
    type requirements.

    Limitation: Automatic type casting will not work for arguments whose datatypes are
    conditionally dependent on other values (i.e., when the datatype requirement appears
    only in the then_branch or else_branch of an If constraint).
    """
    from nvtripy.frontend.constraints.logic import If

    # We perform this operation in two steps:
    # 1. Identify all arguments that are expected to have equal data types.
    # 2. Propagate known data types to all arguments in each equality set.
    expected_equal_dtype: List[Set[str]] = []

    def insert_pair(name1, name2):
        for pair_set in expected_equal_dtype:
            if name1 in pair_set or name2 in pair_set:
                pair_set.update({name1, name2})
                return
        expected_equal_dtype.append({name1, name2})

    known_dtypes: Dict[str, Optional[tp_dtype]] = {}
    arg_map: Dict[str, Any] = {name: value for name, value in merged_args}

    def _is_trusted_dtype_source(value: Any) -> bool:
        """Return True if `value` should be treated as a stable dtype source for autocasting.

        We intentionally treat Python scalar literals (int/float/bool) as *untrusted* sources
        so that expressions like `tensor_f16 * 1.0` will cast the scalar to the tensor dtype
        rather than forcing the tensor to match the scalar's default dtype.
        """
        from nvtripy.frontend.tensor import Tensor

        if isinstance(value, Tensor) or isinstance(value, tp_dtype):
            return True
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return all(_is_trusted_dtype_source(v) for v in value) if len(value) > 0 else False
        return False

    for name, _ in merged_args:

        # If this argument already has a known dtype and is a trusted source, populate it:
        if _is_trusted_dtype_source(arg_map.get(name)):
            try:
                known_dtypes[name] = GetDataType(GetInput(name))(merged_args)
            except Exception:
                pass

        def process_dtype_equality(matched_constraints, input_is_lhs):
            for constraint in matched_constraints:
                expected = constraint.fetcher_or_value if input_is_lhs else constraint.fetcher
                if isinstance(expected, GetDataType):
                    # This might be too restrictive, in which case the assertion can be lifted and the logic here updated.
                    # However, it should generally be the case that input requirements are in terms of the inputs.
                    assert isinstance(
                        expected.value_fetcher, GetInput
                    ), f"Input requirements should only look at inputs"
                    other_name = expected.value_fetcher.name

                    insert_pair(name, other_name)

                    try:
                        known_dtypes[other_name] = expected(merged_args)
                    except Exception:
                        # dtype is not yet known (i.e. might be comparing two inputs with unknown dtypes)
                        pass

                else:
                    known_dtypes[name] = expected(merged_args) if isinstance(expected, Fetcher) else expected

        # NOTE: Because we check for the input on both sides of the equality, we do not need to do another pass over
        # expected_equal_dtype to merge disjoint sets - if we have a transitive equality like:
        #  `a == c and b == d and b == a`, then `a`, `c`, `b` will be immediately added to the same set, which `d` will
        # join when we process `b`.
        # Skip searching inside If constraints to avoid treating conditional checks as type requirements:
        process_dtype_equality(
            input_requirements.find(Equal(GetDataType(GetInput(name)), None), skip_within=If), input_is_lhs=True
        )
        process_dtype_equality(
            input_requirements.find(Equal(None, GetDataType(GetInput(name))), skip_within=If), input_is_lhs=False
        )

    # We do not need to perform validation, as that will happen during constraints checking.
    for dtype_set in expected_equal_dtype:
        # Prefer dtypes coming from trusted sources (tensors / explicit dtypes).
        trusted_names_in_order = [
            n for (n, _) in merged_args if n in dtype_set and _is_trusted_dtype_source(arg_map.get(n))
        ]
        candidate_dtypes = [known_dtypes.get(n) for n in trusted_names_in_order if known_dtypes.get(n) is not None]

        # If there are conflicting trusted dtypes, do not guess.
        known_dtype_in_set: Optional[tp_dtype]
        if len(set(candidate_dtypes)) > 1:
            known_dtype_in_set = None
        else:
            known_dtype_in_set = candidate_dtypes[0] if candidate_dtypes else None

        # dtype might be unknown if the arguments are all non-tensor / untrusted types.
        # In that case, we intentionally do *not* treat inferred scalar literal dtypes (e.g. 1.0 -> float32)
        # as "known" to avoid unnecessary/incorrect casting behavior.
        for name in dtype_set:
            if known_dtype_in_set is None:
                known_dtypes[name] = None
                continue

            # If this argument is an untrusted dtype source (e.g., Python scalar), prefer the
            # trusted dtype chosen for the group even if we previously inferred a dtype for it.
            if not _is_trusted_dtype_source(arg_map.get(name)):
                known_dtypes[name] = known_dtype_in_set
            else:
                known_dtypes.setdefault(name, known_dtype_in_set)

    return known_dtypes


# Performs type conversions if needed. Returns updated values of args, kwargs, and merged args
def convert_input_types(
    func,
    args,
    kwargs,
    merged_args: List[Tuple[str, Any]],
    var_arg_info,
    conversion_targets,
    conversion_preprocess_func,
    shape_likes,
    input_requirements: Constraints,
):
    from nvtripy.common.datatype import bool as tp_bool
    from nvtripy.common.datatype import floating, integer
    from nvtripy.frontend.dimension_size import DimensionSize
    from nvtripy.frontend.ops.cast import cast
    from nvtripy.frontend.ops.utils import tensor_from_shape_like
    from nvtripy.frontend.tensor import Tensor

    if conversion_preprocess_func is not None:
        var_arg_name, var_arg_start_idx = utils.utils.default(var_arg_info, (None, None))
        new_args = conversion_preprocess_func(*args, **kwargs)
        for index in range(len(merged_args)):
            name, _ = merged_args[index]
            if name in new_args:
                if name == var_arg_name:
                    assert var_arg_start_idx is not None
                    merged_args[index] = (name, new_args[name][index - var_arg_start_idx])
                else:
                    merged_args[index] = (name, new_args[name])

    known_datatypes: Dict[str, Optional[tp_dtype]] = {}
    if input_requirements is not None:
        known_datatypes = _find_known_datatypes(merged_args, input_requirements)

    new_args = []
    new_kwargs = {}
    new_merged_args = []
    for index, (name, arg) in enumerate(merged_args):

        def add_arg(arg):
            # need to keep merged args up to date to reuse in the future
            new_merged_args.append((name, arg))
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
                arg = DimensionSize(arg) if type(arg) is int else Tensor(arg)

            _add_column_info(
                arg,
                index,
                name in kwargs,
                len(args),
                func.__name__,
            )

            dtype = None
            if input_requirements is not None:
                dtype = known_datatypes.get(name)

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

    return new_args, new_kwargs, new_merged_args


# Modify the docstring to include constraints
def _update_docstring(func, input_requirements, output_guarantees):
    if not func.__doc__:
        return

    if input_requirements is None and output_guarantees is None:
        return

    indentation = " " * 4
    code_block_index = func.__doc__.find(".. code-block:: python")
    assert code_block_index != -1, f"No code example in docstring for {func.__name__}"

    input_requirements_str = (
        f"\nINPUT REQUIREMENTS:\n{indent(doc_str(input_requirements), indentation)}\n" if input_requirements else ""
    )
    output_guarantees_str = (
        f"\nOUTPUT GUARANTEES:\n{indent(doc_str(output_guarantees), indentation)}\n" if output_guarantees else ""
    )

    func.__doc__ = (
        func.__doc__[:code_block_index]
        + indent(input_requirements_str + output_guarantees_str, indentation)
        + "\n"
        + indentation
        + func.__doc__[code_block_index:]
    )


def interface(
    input_requirements: Optional[Constraints] = None,
    output_guarantees: Optional[Constraints] = None,
    convert_to_tensors: Union[bool, Set[str]] = False,
    conversion_preprocess_func: Optional[Callable] = None,
):
    """
    Decorator for specifying constraints and transformations on front-end inputs. To avoid having to
    layer too many decorators, it is preferable to extend this decorator with further functionality
    than to add and apply further decorators.

    Args:
        input_requirements: A constraints tree that validates function inputs.
            If provided and input validation is enabled, these constraints are checked at runtime.
        output_guarantees: A constraints tree describing guarantees about the function output.
            If provided, these are used for documentation and tooling.
        convert_to_tensors: If False or an empty set, no argument types will be converted.
            If True, all arguments with the `TensorLike` or `ShapeLike` annotations will be
            converted into `Tensor`s or, whenever possible, `DimensionSize`. If the argument
            is a set of argument names, conversions will be done only for those arguments.

            The conversions will attempt safe casts as needed based on `input_requirements`,
            but will raise an exception for lossy casts like float to int (but *not* for, e.g., `float32` to `float16`).

        conversion_preprocess_func: If `convert_to_tensors` is true, this argument is a callback that is
            used to preprocess the arguments before potential conversion. In this case, if provided, the callback
            will be called regardless of whether the decorator performs any conversions.

            The callback will be called with all arguments that were passed to the decorated function and should
            return a dictionary of all updated arguments. For a variadic arg, the dictionary entry for the name
            should have a list of all the updated values.
    """

    def decorator(func):
        from nvtripy.types import ShapeLike, TensorLike

        optimized_input_requirements = optimize_constraints(input_requirements)
        if isinstance(optimized_input_requirements, AlwaysTrue):
            optimized_input_requirements = None

        signature = inspect.signature(func)
        conversion_targets = (
            convert_to_tensors
            if isinstance(convert_to_tensors, Set)
            else {name for name, param in signature.parameters.items() if param.annotation in {TensorLike, ShapeLike}}
        )
        shape_likes = {name for name, param in signature.parameters.items() if param.annotation is ShapeLike}

        # Register constraints for Tripy operators if either side is specified.
        #
        # NOTE: The interface decorator is also used in unit tests and potentially by user code.
        # We only want Tripy's own operators to appear in the global registry that powers
        # public-API validation and integration tests.
        if (input_requirements is not None or output_guarantees is not None) and func.__module__.startswith("nvtripy"):
            OPERATOR_CONSTRAINTS.append(OperatorConstraints(func, input_requirements, output_guarantees))
            _update_docstring(func, input_requirements, output_guarantees)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            merged_args = None
            omitted_default_args = None
            var_arg_info = None

            def get_merged_args():
                nonlocal merged_args, omitted_default_args, var_arg_info
                if merged_args is None:
                    merged_args, omitted_default_args, var_arg_info = utils.utils.merge_function_arguments(
                        func, *args, **kwargs
                    )
                return merged_args, omitted_default_args, var_arg_info

            if convert_to_tensors:
                merged_args, omitted_default_args, var_arg_info = get_merged_args()
                args, kwargs, merged_args = convert_input_types(
                    func,
                    args,
                    kwargs,
                    merged_args,
                    var_arg_info,
                    conversion_targets,
                    conversion_preprocess_func,
                    shape_likes,
                    input_requirements,
                )

            if config.enable_input_validation:
                if optimized_input_requirements is not None:
                    merged_args, omitted_default_args, _ = get_merged_args()
                    # Input validation needs to know values for arguments that were not provided but have default values:
                    result = optimized_input_requirements(merged_args + omitted_default_args)
                    if not result:
                        details = (
                            ["Expected: "]
                            + result.error_details
                            + [f".\n\nNote: Requirements are:\n    {input_requirements}."]
                        )

                        # Include source locations for relevant tensor inputs to make constraint
                        # failures actionable.
                        for name, value in merged_args + omitted_default_args:
                            if hasattr(value, "stack_info"):
                                details.extend([f"\n\nArgument '{name}' was defined here:\n", value])

                        raise_error(
                            f"Invalid inputs for function: '{func.__qualname__}'.",
                            details,
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def constant_fields(field_names: Sequence[str]):
    """
    Marks fields as immutable and disallows them from being changed
    once they have been set the first time.

    Args:
        field_names: The names of fields that should be made immutable.
    """

    def constant_fields_impl(cls: type):
        default_init = cls.__init__

        @functools.wraps(default_init)
        def custom_init(self, *args, **kwargs):
            self.__initialized_fields = set()
            return default_init(self, *args, **kwargs)

        default_setattr = cls.__setattr__

        @functools.wraps(default_setattr)
        def custom_setattr(self, name, value):
            if name == "__initialized_fields":
                return object.__setattr__(self, name, value)

            if name in field_names:
                if name in self.__initialized_fields:
                    raise_error(f"Field: '{name}' of class: '{cls.__qualname__}' is immutable!")
                self.__initialized_fields.add(name)

            return default_setattr(self, name, value)

        cls.__init__ = custom_init
        cls.__setattr__ = custom_setattr
        return cls

    return constant_fields_impl
