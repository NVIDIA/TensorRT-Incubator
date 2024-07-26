
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import functools
import inspect
from collections import OrderedDict, defaultdict
from textwrap import dedent
from typing import Any, Callable, Dict, List, Tuple

from dataclasses import dataclass


@dataclass
class AnnotationInfo:
    type_info: type
    optional: bool
    kind: Any  # Uses inspect.Parameter.<kind>


class FuncOverload:
    def __init__(self, func):
        self.func = func
        # Cache evaluated type annotations.
        # We *cannot* populate this at `__init__` time since that will be evaluated when the function
        # is first defined, at which point the required types in the annotations may not be accessible.
        # Instead, we populate this the first time the function is called.
        self.annotations = None

    def __str__(self) -> str:
        from tripy.utils.utils import code_pretty_str

        _, lineno = inspect.getsourcelines(self.func)
        # For long source code, just keep the first few lines.
        source_lines = inspect.getsource(self.func).splitlines()
        CONTEXT_LEN = 3
        if len(source_lines) > CONTEXT_LEN + 1:  # + 1 for the '...' line we add
            source_lines = source_lines[:CONTEXT_LEN] + ["    ..."]
        source_code = "\n".join(source_lines)
        pretty_code = code_pretty_str(source_code, inspect.getsourcefile(self.func), lineno, self.func.__name__)
        return pretty_code + "\n"

    def _get_annotations(self):
        if self.annotations is None:
            # Maps parameter names to their type annotations and a boolean indicating whether they are optional.
            self.annotations: Dict[str, AnnotationInfo] = OrderedDict()
            signature = inspect.signature(self.func)
            for name, param in signature.parameters.items():
                if name == "self":
                    # Not likely to pass in the wrong `self` parameter, so we
                    # don't require an annotation for it.
                    annotation = Any
                else:
                    assert (param.annotation and param.annotation is not signature.empty) or param.kind in {
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    }, f"Non-variadic function parameters must have type annotations, but parameter: '{name}' of function: '{self.func.__name__}' has no type annotation!"
                    annotation = param.annotation
                    # In cases where a type is not available at the time of function definition, the type
                    # annotation may be provided as a string. Since we need the actual type, we just
                    # eval it here.
                    if isinstance(annotation, str):
                        try:
                            # Import tripy so we can evaluate types from within tripy.
                            import tripy

                            annotation = eval(annotation)
                        except Exception as e:
                            raise NameError(
                                f"Error while evaluating type annotation: '{annotation}' for parameter: '{name}' of function: '{self.func.__name__}'."
                                f"\nNote: Error was: {e}"
                            )

                self.annotations[name] = AnnotationInfo(annotation, param.default is not signature.empty, param.kind)

        return self.annotations

    def matches_arg_types(self, args, kwargs) -> "Result":
        from tripy.utils.result import Result

        def matches_type(name: str, annotation: type, arg: Any):
            # In cases where a type is not available at the time of function definition, the type
            # annotation may be provided as a string. Since we need the actual type, we just
            # eval it here.
            if isinstance(annotation, str):
                try:
                    annotation = eval(annotation)
                except Exception as e:
                    raise NameError(
                        f"Error while evaluating type annotation: '{annotation}' for parameter: '{name}' of function: '{self.func.__name__}'."
                        f"\nNote: Error was: {e}"
                    )

            try:
                return isinstance(arg, annotation)
            except TypeError:
                # When the type annotation includes a subscripted generic (e.g. Tuple[int]), isinstance
                # does not work. We could introduce more advanced type checking using `typing.get_origin` and `typing.get_args`
                # but for now, we don't support overloading on generic types.
                return True

        annotations = self._get_annotations()

        # Check if we have too many positional arguments. We can only do this if there isn't a variadic positional argument.
        if not any(annotation.kind == inspect.Parameter.VAR_POSITIONAL for annotation in annotations.values()) and len(
            args
        ) > len(annotations):
            return Result.err(
                [f"Function expects {len(annotations)} parameters, but {len(args)} arguments were provided."],
            )

        for (name, annotation), arg in zip(annotations.items(), args):
            if not matches_type(name, annotation.type_info, arg):
                return Result.err(
                    [
                        f"For parameter: '{name}', expected an instance of type: "
                        f"'{annotation.type_info.__qualname__}' but got argument of type: '{type(arg).__qualname__}'."
                    ],
                )

        for name, arg in kwargs.items():
            if name in annotations:
                typ = annotations[name].type_info
                if not matches_type(name, typ, arg):
                    return Result.err(
                        [
                            f"For parameter: '{name}', expected an instance of type: "
                            f"'{typ.__qualname__}' but got argument of type: '{type(arg).__qualname__}'."
                        ],
                    )
            elif not any(annotation.kind == inspect.Parameter.VAR_KEYWORD for annotation in annotations.values()):
                # We can only validate the names of arguments if the function does not accept variadic kwargs
                return Result.err(
                    [
                        f"Parameter: '{name}' is not valid for this function. "
                        f"Note: This function takes the following parameters: {list(annotations.keys())}"
                    ],
                )

        # Check if all required arguments are given. We do so by stripping out arguments
        # and then checking if any of the remaining ones are required.
        # We do this only after we know that all provided args/kwargs are valid.
        # Don't count variadic arguments here since they will always be "missing".
        missing_arg_dict = {
            name: annotation
            for name, annotation in annotations.items()
            if annotation.kind not in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]
        }

        arg_names = list(missing_arg_dict.keys())[: len(args)]
        kwarg_names = list(kwargs.keys())

        for name in arg_names + kwarg_names:
            # Names might not be present in the initial missing_arg_dict (which is the entire function signature) in case of e.g. variadic kwargs
            if name in missing_arg_dict:
                del missing_arg_dict[name]

        missing_required_args = [name for name, annotation in missing_arg_dict.items() if not annotation.optional]
        if missing_required_args:
            return Result.err([f"Some required arguments were not provided: {missing_required_args}"])

        return Result.ok()

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class FunctionRegistry(dict):
    """
    Maps function names to function implementations.

    This class supports limited function overloading; it can dispatch to overloads whose
    parameter names differ or which have different parameter types (with caveats - see below).

    NOTE: Currently, generic types (e.g. `List[int]`) are *not* supported. Thus, functions
    that differ only by parameter type must differ on at least one simple type in order for
    overload resolution to work. A way around this is to use different parameter names and
    require the user to use keyword arguments when calling the function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overloads: Dict[str, List[Callable]] = defaultdict(list)

    # NOTE: If you change this signature, also update `stack_info.py` - it currently relies on getting `key` to determine function names.
    def find_overload(self, key: str, args: List, kwargs: Dict) -> Callable:
        # NOTE: We could introduce a fast path when len(overloads) == 1, but it seems worth the overhead
        # to have the type checks and nice error messages. Note that this overhead will not be a factor when we compile.

        def raise_overload_error(msg, candidate_overloads, mismatch_reasons=None, extra_info=""):
            arg_type_strs = []
            arg_type_strs.extend(type(arg).__qualname__ for arg in args)
            arg_type_strs.extend(f"{name}={type(value).__qualname__}" for name, value in kwargs.items())

            overloads_error = []
            for index, candidate in enumerate(candidate_overloads):
                overloads_error.append(f"{str(candidate)}\n")
                if mismatch_reasons:
                    # This path should only be entered when all the overloads are mismatched since we don't provide `mismatch_reasons` otherwise.
                    assert len(mismatch_reasons) == len(candidate_overloads)
                    overloads_error.append("Not a valid overload because: ")
                    overloads_error.extend(mismatch_reasons[index])
                    overloads_error.append("\n\n")

            from tripy.common.exception import raise_error

            raise_error(
                f"{msg} for function: '{key}'.",
                details=[
                    f"Note: Argument types were: [{', '.join(arg_type_strs)}].\nCandidate overloads were:\n\n",
                    *overloads_error,
                    extra_info,
                ],
            )

        matched_overloads = []
        mismatch_reasons = []
        for overload in self.overloads[key]:
            matched = overload.matches_arg_types(args, kwargs)
            if matched:
                matched_overloads.append(overload)
            else:
                mismatch_reasons.append(matched.error_details)

        if len(matched_overloads) > 1:
            raise_overload_error(
                "Ambiguous overload",
                matched_overloads,
                extra_info="Hint: Try using keyword arguments to help disambiguate between overloads.",
            )
        elif matched_overloads:
            return matched_overloads[0]

        raise_overload_error("Could not find an implementation", self.overloads[key], mismatch_reasons)

    def __call__(self, key: Any):
        """
        Registers a function with this function registry.
        This function allows instances of the class to be used as decorators.

        Args:
            key: The key under which to register the function.
        """

        def impl(func):
            # Cannot dispatch on properties, so we just use func directly.
            # Otherwise, we generate a dispatch function which will dispatch to the appropriate overload
            # based on argument types.
            if isinstance(func, property):
                self[key] = func
            else:
                self.overloads[key].append(FuncOverload(func))
                # The dispatch function needs to look and feel like the underlying function to make docs
                # work correctly. When there are multiple overloads, we concatenate the docstrings together
                # for the dispatch function.
                if key not in self:

                    # NOTE: If you change this signature, also update `stack_info.py` - it currently relies on the `wrapper` name.
                    @functools.wraps(func)
                    def wrapper(*args, **kwargs):
                        return self.find_overload(key, args, kwargs)(*args, **kwargs)

                    self[key] = wrapper
                else:
                    # By deleting __wrapped__, we undo parts of what `functools.wraps` does.
                    # This allows us to omit signature information in the docs and prepend our
                    # own to the docstrings.
                    del self[key].__wrapped__

                    # Add a special attribute to the function so we know it's an overload dispatcher
                    self[key].is_overload_dispatcher = True

                    def prepend_signature_to_docstring(f):
                        if not f.__doc__:
                            return ""

                        signature = inspect.signature(f)

                        def str_from_annotation(annotation):
                            if isinstance(annotation, str):
                                ret = annotation
                            else:
                                ret = annotation.__qualname__
                            return f":class:`{ret}`"

                        def make_param_str(param):
                            param_str = f"*{param.name}*: {str_from_annotation(param.annotation)}"
                            if param.default != signature.empty:
                                param_str += f" = {param.default}"
                            return param_str

                        sig_str = f"> **{key}** ({', '.join(make_param_str(param) for param in signature.parameters.values() if param.name != 'self')}) -> "

                        if signature.return_annotation != signature.empty:
                            sig_str += f"{str_from_annotation(signature.return_annotation)}"
                        else:
                            sig_str += "None"

                        section_divider = "-" * 10
                        return (f"""\n\n{section_divider}\n\n{sig_str}\n{dedent(f.__doc__)}""").strip()

                    # The first time we add an overload, we need to retroactively process the existing docstring
                    # to add signature information.
                    if len(self.overloads[key]) == 2:
                        self[key].__doc__ = (
                            "*This function has multiple overloads:*\n\n"
                            + prepend_signature_to_docstring(self.overloads[key][0].func)
                        )

                    self[key].__doc__ += "\n\n" + prepend_signature_to_docstring(func)

            return self[key]

        return impl
