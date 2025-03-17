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
from collections.abc import Callable as ABCCallable
from collections.abc import Sequence as ABCSequence
from typing import Any, Dict, ForwardRef, Union, get_args, get_origin


# Returns either the name of the object if it has one, or the name of its type.
def obj_name_or_type_name(obj):
    try:
        return obj.__name__
    except AttributeError:
        return type(obj).__name__


def _get_type_name(typ):
    # Attach module name if possible
    module_name = ""
    try:
        module_name = typ.__module__ + "."
    except AttributeError:
        pass
    else:
        # Modify prefix for built-in types or Tripy types.
        # If we include modules for Tripy, they will include all submodules, which can be confusing
        # e.g. Tensor will be something like "nvtripy.frontend.tensor.Tensor"
        if module_name.startswith("nvtripy"):
            module_name = "nvtripy."
        if module_name.startswith("builtins"):
            module_name = ""

    return module_name + typ.__qualname__


def str_from_type_annotation(annotation, postprocess_annotation=None):
    def get_name(obj):
        try:
            # Python 3.9 does not have a __name__ attribute for annotations.
            return obj.__name__
        except AttributeError:
            return obj._name

    postprocess_annotation = postprocess_annotation or (lambda x: x)

    if annotation is type(None):
        return postprocess_annotation("None")

    if isinstance(annotation, str):
        return postprocess_annotation(annotation)

    if get_origin(annotation) is Union:
        types = list(get_args(annotation))
        return " | ".join(str_from_type_annotation(typ, postprocess_annotation) for typ in types)

    if get_origin(annotation) in {ABCSequence, list, tuple}:
        types = get_args(annotation)
        return f"{get_name(annotation)}[{', '.join(str_from_type_annotation(typ, postprocess_annotation) for typ in types)}]"

    if get_origin(annotation) is ABCCallable:
        params, ret = get_args(annotation)
        return f"{get_name(annotation)}[[{', '.join(map(str_from_type_annotation, params))}], {str_from_type_annotation(ret)}]"

    if get_origin(annotation) is dict:
        key_annotations, value_annotations = get_args(annotation)
        return f"Dict[{str_from_type_annotation( key_annotations)}, {str_from_type_annotation( value_annotations)}]"

    if isinstance(annotation, ForwardRef):
        return postprocess_annotation(str(annotation.__forward_arg__))

    # typing module annotations are likely to be better when pretty-printed due to including subscripts
    return postprocess_annotation(str(annotation) if annotation.__module__ == "typing" else _get_type_name(annotation))


def type_str_from_arg(arg: Any) -> str:
    # it is more useful to report more detailed types for sequences/tuples in error messages
    from typing import List, Tuple

    if isinstance(arg, List):
        if len(arg) == 0:
            return "List"
        arg_types = sorted({type_str_from_arg(member) for member in arg})
        return f"List[{' | '.join(arg_types)}]"
    elif isinstance(arg, Tuple):
        return f"Tuple[{', '.join(map(type_str_from_arg, arg))}]"
    elif isinstance(arg, Dict):
        if len(arg) == 0:
            return "Dict"
        key_types = sorted({type_str_from_arg(key) for key in arg.keys()})
        value_types = sorted({type_str_from_arg(value) for value in arg.values()})
        return f"Dict[{' | '.join(key_types)}, {' | '.join(value_types)}]"

    return _get_type_name(type(arg))
