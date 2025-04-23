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

import functools
import inspect
from collections import OrderedDict, defaultdict
from collections.abc import Callable as ABCCallable
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass
from textwrap import dedent, indent
from typing import Any, Callable, Dict, ForwardRef, List, Optional, Sequence, Union, get_args, get_origin

from nvtripy.utils.types import str_from_type_annotation, type_str_from_arg


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
        self._annotations = None

    def __str__(self) -> str:
        from nvtripy.utils.utils import code_pretty_str

        lines, lineno = inspect.getsourcelines(self.func)

        func_def_start_index = 0
        func_def_end_index = 0
        for index, line in enumerate(lines):
            if line.strip().startswith("def"):
                func_def_start_index = index
            if "):" in line or ") ->" in line:
                func_def_end_index = index
                break

        func_def_end_index = max(func_def_start_index, func_def_end_index)
        lines = lines[func_def_start_index : func_def_end_index + 1]
        source_code = "\n".join(map(lambda line: line.rstrip(), lines))
        pretty_code = code_pretty_str(source_code, inspect.getsourcefile(self.func), lineno, self.func.__name__)
        return pretty_code + "\n"

    def _get_annotations(self):
        if self._annotations is not None:
            return self._annotations

        # Maps parameter names to their type annotations and a boolean indicating whether they are optional.
        self._annotations: Dict[str, AnnotationInfo] = OrderedDict()
        signature = inspect.signature(self.func)
        for name, param in signature.parameters.items():
            if name == "self" or name == "cls":
                # Not likely to pass in the wrong `self` or `cls` parameter, so we
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
                        # Import nvtripy so we can evaluate types from within nvtripy.
                        import nvtripy

                        annotation = eval(annotation)
                    except Exception as e:
                        raise NameError(
                            f"Error while evaluating type annotation: '{annotation}' for parameter: '{name}' of function: '{self.func.__name__}'."
                            f"\nNote: Error was: {e}"
                        )

            self._annotations[name] = AnnotationInfo(annotation, param.default is not signature.empty, param.kind)

        return self._annotations

    def matches_arg_types(self, args, kwargs) -> "Result":
        from itertools import chain

        from nvtripy.utils.result import Result

        def matches_type(name: str, annotation: type, arg: Any) -> bool:
            if annotation is Any:
                return True

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

            # can add more cases, prioritizing the common ones
            if get_origin(annotation) is Union:
                return any(map(lambda type_arg: matches_type(name, type_arg, arg), get_args(annotation)))

            # note: get_origin for typing.Sequence normalizes it into collections.abc.Sequence, see spec for get_origin
            if get_origin(annotation) in {ABCSequence, list}:
                # in the context of Tripy, it does not make sense to consider strings as sequences
                if not isinstance(arg, Sequence) or isinstance(arg, str):
                    return False
                seq_arg = get_args(annotation)
                if seq_arg and len(arg) > 0:
                    assert len(seq_arg) == 1
                    return all(map(lambda member: matches_type(name, seq_arg[0], member), arg))
                return True

            if get_origin(annotation) is ABCCallable:
                # Note: Callables passed in may not have type annotations, so it there is no general way
                # to validate that they accept the right types.
                return callable(arg)

            if get_origin(annotation) is Union and annotation._name == "Optional":
                return arg is None or matches_type(arg, get_args(annotation)[0])

            if get_origin(annotation) is dict:
                if not isinstance(arg, Dict):
                    return False

                key_annotation, value_annotation = get_args(annotation)
                return all(
                    matches_type(name, key_annotation, key) and matches_type(name, value_annotation, value)
                    for key, value in arg.items()
                )

            if get_origin(annotation) is tuple:
                if not isinstance(arg, tuple):
                    return False

                tuple_annotations = get_args(annotation)
                if len(tuple_annotations) != len(arg):
                    return False
                return all(
                    matches_type(name, tuple_annotation, tuple_arg)
                    for tuple_annotation, tuple_arg in zip(tuple_annotations, arg)
                )

            # Forward references can be used for recursive type definitions. Warning: Has the potential for infinite looping if there is no base case!
            if isinstance(annotation, ForwardRef):
                # NOTE: We need this import in case the annotation references nvtripy
                import nvtripy

                return matches_type(name, eval(annotation.__forward_arg__), arg)

            return isinstance(arg, annotation)

        annotations = self._get_annotations()

        # Check if we have too many positional arguments. We can only do this if there isn't a variadic positional argument.
        annotation_items = list(annotations.items())
        variadic_idx = None
        for idx, (_, annotation) in enumerate(annotation_items):
            # there can only be at most one variadic arg and it must come after all positional args and before keyword-only args
            if annotation.kind == inspect.Parameter.VAR_POSITIONAL:
                variadic_idx = idx
                break

        if variadic_idx is None and len(args) > len(annotations):
            return Result.err(
                [f"Function expects {len(annotations)} parameters, but {len(args)} arguments were provided."],
            )

        # If there is a variadic positional arg, we can copy the final annotation for the remaining args.
        # Keyword-only args (only possible with a variadic arg) will appear in kwargs and don't need to be checked here.
        if variadic_idx is not None:
            positional_args_to_check = chain(
                zip(annotation_items[:variadic_idx], args),
                map(lambda arg: (annotation_items[variadic_idx], arg), args[len(annotations) - 1 :]),
            )
        else:
            positional_args_to_check = zip(annotation_items, args)

        for (name, annotation), arg in positional_args_to_check:
            if not matches_type(name, annotation.type_info, arg):
                return Result.err(
                    [
                        f"For parameter: '{name}', expected an instance of type: '{str_from_type_annotation(annotation.type_info)}' "
                        f"but got argument of type: '{type_str_from_arg(arg)}'."
                    ]
                )

        for name, arg in kwargs.items():
            if name in annotations:
                typ = annotations[name].type_info
                if not matches_type(name, typ, arg):
                    return Result.err(
                        [
                            f"For parameter: '{name}', expected an instance of type: '{str_from_type_annotation(typ)}' "
                            f"but got argument of type: '{type_str_from_arg(arg)}'."
                        ]
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

            from nvtripy.common.exception import raise_error

            raise_error(
                f"{msg} for function: '{key}'.",
                details=[
                    f"Candidate overloads were:\n\n",
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

    def __call__(self, key: Any, bypass_dispatch: Optional[Union[bool, Sequence[str]]] = None):
        """
        Registers a function or class with this function registry.
        This function allows instances of the class to be used as decorators.
        If the decorator is applied to a class, all _documented_ methods of the class will be added into the registry as well,
        with the method name appended to the key in the format "key.{method name}".

        Args:
            key: The key under which to register the function.

            bypass_dispatch: Has no effect if the value is None or False.
               If the decorator is applied to a function and the value is true, this will add the decorated function to the registry but does not
               attempt to dispatch overloads. This option avoids overhead but does not perform automatic type-checking or allow for choosing between overloads.
               If the decorator is applied to a class and the value is true, dispatch will be bypassed for all methods.
               If the decorator is applied to a class and the value is a list of method names, dispatch will be bypassed for the listed methods only.
        """
        bypass_dispatch = bypass_dispatch or False

        def impl(func):
            # Cannot dispatch on properties, so we just use func directly.
            # Otherwise, we generate a dispatch function which will dispatch to the appropriate overload
            # based on argument types.
            if isinstance(func, property):
                self[key] = func
            # For classes, we apply the wrapper to all methods.
            elif inspect.isclass(func):
                # Ignore non-public properties and functions and those not defined in the class (we will use the presence of a docstring as a proxy for that).
                # It does not suffice to check just that the method is inherited because some decorators like @dataclass add methods
                # that are not documented or annotated and do not use inheritance to do so.
                for name, member in inspect.getmembers(
                    func, predicate=lambda m: inspect.isfunction(m) and m.__doc__ is not None
                ):
                    bypass_func = bypass_dispatch if isinstance(bypass_dispatch, bool) else name in bypass_dispatch
                    setattr(func, name, self(f"{key}.{name}", bypass_dispatch=bypass_func)(member))
                self[key] = func
            else:
                assert not (
                    bypass_dispatch and key in self
                ), f"Attempting to add key '{key}' into a function registry with dispatch disabled, but there is already an overload present."

                self.overloads[key].append(FuncOverload(func))

                if bypass_dispatch:
                    self[key] = func
                    self[key].is_dispatch_disabled = True  # added to avoid later adding an overload
                    return func

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
                    assert not hasattr(
                        self[key], "is_dispatch_disabled"
                    ), f"Dispatch was disabled for key '{key}', but a second overload was registered for it."

                    # By deleting __wrapped__, we undo parts of what `functools.wraps` does.
                    # This allows us to omit signature information in the docs and prepend our
                    # own to the docstrings.
                    del self[key].__wrapped__

                    # Add a special attribute to the function so we know it's an overload dispatcher
                    self[key].is_overload_dispatcher = True

                    def prepend_signature_to_docstring(f):
                        if not f.__doc__:
                            return ""

                        roles = ""

                        def add_role(name, *additional_classes):
                            nonlocal roles

                            classes = [name] + list(additional_classes)
                            roles += f".. role:: {name}\n    :class: {' '.join(classes)}\n"

                        add_role("sig-prename", "descclassname")
                        add_role("sig-name", "descname")

                        # We cannot use `FuncOverload._get_annotations()` here because it is too early to be able
                        # to import nvtripy to evaluate annotations.
                        signature = inspect.signature(f)

                        postprocess_annotation = lambda annotation: (
                            f":class:`{annotation}`" if annotation.startswith("nvtripy.") else annotation
                        )

                        def make_param_str(param):
                            param_str = (
                                f"{param.name}: {str_from_type_annotation(param.annotation, postprocess_annotation)}"
                            )
                            if param.default != signature.empty:
                                param_str += f" = {param.default}"
                            return param_str

                        sig_str = rf":sig-prename:`nvtripy`\ .\ :sig-name:`{key}`\ ({', '.join(make_param_str(param) for param in signature.parameters.values() if param.name != 'self')}) -> "

                        if signature.return_annotation != signature.empty:
                            sig_str += (
                                f"{str_from_type_annotation(signature.return_annotation, postprocess_annotation)}"
                            )
                        else:
                            sig_str += "None"

                        section_divider = "-" * 10
                        indent_prefix = " " * 4
                        # We add a special `func-overload-sig` class here so we can correct the documentation
                        # styling for signatures of overloaded functions.
                        overload_doc = (
                            f"""\n\n{section_divider}\n\n{dedent(roles).strip()}\n\n.. container:: func-overload-sig sig sig-object py\n\n{indent(sig_str, indent_prefix)}\n{dedent(f.__doc__)}"""
                        ).strip()
                        return overload_doc

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
