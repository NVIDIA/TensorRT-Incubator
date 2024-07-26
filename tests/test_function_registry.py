
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

import inspect
from textwrap import dedent
from typing import Any, Dict, List

import pytest

from tripy import TripyException
from tripy.function_registry import AnnotationInfo, FunctionRegistry


@pytest.fixture()
def registry():
    yield FunctionRegistry()


@pytest.fixture
def int_float_registry(registry):
    @registry("transform")
    def transform_int(a: "int"):
        return a + 1

    @registry("transform")
    def transform_float(a: "float"):
        return a - 1

    yield registry


class TestFunctionRegistry:
    def test_basic_registration(self, registry):
        @registry("increment")
        def func(a: int):
            return a + 1

        assert "increment" in registry
        assert registry["increment"](1) == 2

    def test_overloading_real_type_annotations(self, int_float_registry):
        assert int_float_registry["transform"](1) == 2
        assert int_float_registry["transform"](1.0) == 0.0

    def test_overloading_string_annotations(self, int_float_registry):
        assert int_float_registry["transform"](1) == 2
        assert int_float_registry["transform"](1.0) == 0.0

    def test_error_on_missing_overload(self, int_float_registry):
        with pytest.raises(
            TripyException,
            match=dedent(
                rf"""
            Could not find an implementation for function: 'transform'.
                Note: Argument types were: \[str\].
                Candidate overloads were:

                --> {__file__}:[0-9]+
                    |
                 [0-9]+ |     \@registry\("transform"\)
                 [0-9]+ |     def transform_int\(a: "int"\):
                 [0-9]+ |         return a \+ 1
                    |

                Not a valid overload because: For parameter: 'a', expected an instance of type: 'int' but got argument of type: 'str'.

                --> {__file__}:[0-9]+
                    |
                 [0-9]+ |     \@registry\("transform"\)
                 [0-9]+ |     def transform_float\(a: "float"\):
                 [0-9]+ |         return a \- 1
                    |

                Not a valid overload because: For parameter: 'a', expected an instance of type: 'float' but got argument of type: 'str'.
            """
            ).strip(),
        ):
            int_float_registry["transform"]("hi")

    def test_error_when_kwargs_wrong(self, registry):
        @registry("test")
        def func(a: int, b: int, c: int):
            return a + b + c

        with pytest.raises(
            TripyException,
            match=dedent(
                rf"""
            Could not find an implementation for function: 'test'.
                Note: Argument types were: \[int, b=int, c=float\].
                Candidate overloads were:

                --> {__file__}:[0-9]+
                    |
                 [0-9]+ |         \@registry\("test"\)
                 [0-9]+ |         def func\(a: int, b: int, c: int\):
                 [0-9]+ |             return a \+ b \+ c
                    |

                Not a valid overload because: For parameter: 'c', expected an instance of type: 'int' but got argument of type: 'float'.
                """
            ).strip(),
        ):
            registry["test"](1, b=2, c=3.0)

    def test_invalid_string_annotation_fails_gracefully(self, registry):
        @registry("test")
        def func(a: "not_a_real_type"):
            pass

        with pytest.raises(
            NameError,
            match="Error while evaluating type annotation: 'not_a_real_type' for parameter: 'a' of function: 'func'."
            "\nNote: Error was: name 'not_a_real_type' is not defined",
        ):
            registry["test"](None)

    def test_overload_different_arg_names(self, registry):
        @registry("test")
        def func(a: int):
            return a + 1

        @registry("test")
        def func(b: int):
            return b - 1

        assert registry["test"](a=0) == 1
        assert registry["test"](b=0) == -1

    def test_overload_different_number_of_args(self, registry):
        @registry("test")
        def func(a: int):
            return a + 1

        @registry("test")
        def func(b: int, c: int):
            return b - 1

        assert registry["test"](0) == 1
        assert registry["test"](0, 0) == -1

    def test_ambiguous_overload_raises_error(self, registry):
        @registry("test")
        def func(a: int):
            return a + 1

        @registry("test")
        def func(b: int):
            return b - 1

        with pytest.raises(
            TripyException,
            match=dedent(
                rf"""
                Ambiguous overload for function: 'test'.
                    Note: Argument types were: \[int\].
                    Candidate overloads were:

                    --> {__file__}:[0-9]+
                        |
                    [0-9]+ |         \@registry\("test"\)
                    [0-9]+ |         def func\(a: int\):
                    [0-9]+ |             return a \+ 1
                        |

                    --> {__file__}:[0-9]+
                        |
                    [0-9]+ |         \@registry\("test"\)
                    [0-9]+ |         def func\(b: int\):
                    [0-9]+ |             return b \- 1
                        |
                """
            ).strip(),
        ):
            assert registry["test"](0) == 1

    def test_missing_arguments_gives_sane_error(self, registry):
        @registry("test")
        def func(a: int, b: int):
            return a + b

        with pytest.raises(TripyException, match="Some required arguments were not provided: \['a'\]"):
            registry["test"](b=0)

    def test_func_overload_caches_signature(self, registry):
        @registry("test")
        def func(a: int):
            return a + 1

        func_overload = registry.overloads["test"][0]

        assert not func_overload.annotations
        assert registry["test"](0) == 1
        assert func_overload.annotations
        assert func_overload.annotations["a"] == AnnotationInfo(int, False, inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def test_doc_of_non_overloaded_func(self, registry):
        # When there is no overload, the registry function should
        # use the docstring as-is
        @registry("test")
        def func():
            """
            An example docstring.
            """
            pass

        assert registry["test"].__doc__.strip() == "An example docstring."

    def test_doc_of_overloaded_func(self, registry):
        @registry("test")
        def func(a: int):
            """
            This func takes an int.
            """
            pass

        @registry("test")
        def func(a: float):
            """
            This func takes a float.
            """
            pass

        print(registry["test"].__doc__)
        assert (
            registry["test"].__doc__
            == dedent(
                """
                *This function has multiple overloads:*

                ----------

                > **test** (*a*: :class:`int`) -> None

                This func takes an int.

                ----------

                > **test** (*a*: :class:`float`) -> None

                This func takes a float.
                """
            ).strip()
        )

    def test_variadic_positional_args(self, registry):
        @registry("test")
        def func(*args: List[Any]):
            return sum(args)

        assert registry["test"](1.0, 2.0, 3.0) == 6.0

    def test_variadic_keyword_args(self, registry):
        @registry("test")
        def func(**kwargs: Dict[str, Any]):
            return sum(kwargs.values())

        assert registry["test"](a=1.0, b=2.0, c=3.0) == 6.0
