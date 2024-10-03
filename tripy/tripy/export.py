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
from dataclasses import dataclass
from typing import List, Optional, Any
from types import ModuleType
from textwrap import dedent
from tripy.function_registry import FunctionRegistry


@dataclass
class PublicAPI:
    obj: Any
    qualname: str
    """The fully qualified name of the API"""
    document_under: str = ""
    autodoc_options: Optional[List[str]] = None


# This is used for testing/documentation purposes.
PUBLIC_APIS: List[PublicAPI] = []


PUBLIC_API_FUNCTION_REGISTRY = FunctionRegistry()


def public_api(
    document_under: str = "",
    autodoc_options: Optional[List[str]] = None,
    module: ModuleType = None,
    symbol: str = None,
    doc: str = None,
):
    """
    Decorator that exports a function/class to the public API under the top-level module and
    controls how it is documented.

    Args:
        document_under: A forward-slash-separated path describing the hierarchy under
            which to document this API.
            For example, providing ``"tensor/initialization"`` would create a directory
            structure like:

            - tensor/
                - initialization/
                    - index.rst
                    - <decorated_function_name>.rst

            This can also be used to target specific `.rst` files directly. Any APIs targeting
            the same `.rst` file will render on the same page in the final docs.

            When targeting an `index.rst` file, the contents of the API will precede the contents
            of the index file.

            You may *not* target the root `index.rst` file.

        autodoc_options: Autodoc options to apply to the documented API.
            For example: ``[":special-members:"]``.

        module: The module under which to export this public API. Defaults to the top-level Tripy module.

        symbol: The name of the symbol, if different from ``__name__``.

        doc: Optional docstring. This is useful in cases where the docstring cannot be provided as normal.
            For example, global variables sometimes don't register docstrings correctly.
    """
    assert not autodoc_options or (
        ":no-members:" not in autodoc_options or ":no-special-members:" in autodoc_options
    ), "Because of how our conf.py file is set up, you must include :no-special-members: when using the :no-members: option!"

    def export_impl(obj):
        nonlocal module, symbol
        import tripy

        if doc is not None:
            obj.__doc__ = dedent(doc)

        module = module or tripy

        symbol = symbol or obj.__name__
        # Leverage the function registry to provide type checking and function overloading capabilities.
        if inspect.isfunction(obj):
            obj = PUBLIC_API_FUNCTION_REGISTRY(symbol)(obj)

        qualname = f"{module.__name__}.{symbol}"
        if inspect.ismodule(obj):
            qualname = symbol

        PUBLIC_APIS.append(PublicAPI(obj, qualname, document_under, autodoc_options))

        if not hasattr(module, "__all__"):
            module.__all__ = []

        module.__all__.append(symbol)
        setattr(module, symbol, obj)

        return obj

    return export_impl
