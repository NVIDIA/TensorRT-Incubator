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
from textwrap import dedent
from typing import Any, Dict, List, Sequence, Union, Optional

import pytest
import torch
from tests import helper

import tripy as tp
from tripy import TripyException
from tripy.function_registry import AnnotationInfo, FunctionRegistry, type_str_from_arg, str_from_type_annotation


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
        # Note presence of ANSI color codes. Also note that the last | has a space after it
        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Could not find an implementation for function: 'transform'\.
                Candidate overloads were:

                --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mtransform_int\(\)\x1b\[0m
                      \|
                   [0-9]+ \|     def transform_int\(a: \"int\"\):
                   [0-9]+ \|     \.\.\.
                      \|\s

                Not a valid overload because: For parameter: 'a', expected an instance of type: 'int' but got argument of type: 'str'\.

                --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mtransform_float\(\)\x1b\[0m
                      \|
                   [0-9]+ \|     def transform_float\(a: "float"\):
                   [0-9]+ \|     \.\.\.
                      \|\s

                Not a valid overload because: For parameter: 'a', expected an instance of type: 'float' but got argument of type: 'str'\.
            """
            ).strip(),
        ):
            int_float_registry["transform"]("hi")

    def test_error_when_kwargs_wrong(self, registry):
        @registry("test")
        def func(a: int, b: int, c: int):
            return a + b + c

        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Could not find an implementation for function: 'test'.
                Candidate overloads were:

                --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mfunc\(\)\x1b\[0m
                      \|
                   [0-9]+ \|         def func\(a: int, b: int, c: int\):
                   [0-9]+ \|     \.\.\.
                      \|\s

                Not a valid overload because: For parameter: 'c', expected an instance of type: 'int' but got argument of type: 'float'\.
                """
            ).strip(),
        ):
            registry["test"](1, b=2, c=3.0)

    def test_invalid_string_annotation_fails_gracefully(self, registry):
        @registry("test")
        def func(a: "not_a_real_type"):
            pass

        with helper.raises(
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

    def test_register_class(self, registry):
        @registry("class")
        class C:
            def f1(self, a: int):
                """
                Need to have a docstring or else it won't be registered."
                """
                return a + 1

            def f2(self, a: int):
                """
                Obligatory docstring.
                """
                return a - 1

        inst_c = C()
        assert registry["class.f1"](inst_c, 1) == 2
        assert registry["class.f2"](inst_c, 1) == 0

    def test_ambiguous_overload_raises_error(self, registry):
        @registry("test")
        def func(a: int):
            return a + 1

        @registry("test")
        def func(b: int):
            return b - 1

        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
                Ambiguous overload for function: 'test'.
                    Candidate overloads were:

                    --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mfunc\(\)\x1b\[0m
                          \|
                      [0-9]+ \|         def func\(a: int\):
                      [0-9]+ \|     \.\.\.
                          \|\s

                    --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mfunc\(\)\x1b\[0m
                          \|
                      [0-9]+ \|         def func\(b: int\):
                      [0-9]+ \|     \.\.\.
                          \|\s
                """
            ).strip(),
        ):
            assert registry["test"](0) == 1

    def test_missing_arguments_gives_sane_error(self, registry):
        @registry("test")
        def func(a: int, b: int):
            return a + b

        with helper.raises(TripyException, match=r"Some required arguments were not provided: \['a'\]"):
            registry["test"](b=0)

    def test_func_overload_caches_signature(self, registry):
        @registry("test")
        def func(a: int):
            return a + 1

        func_overload = registry.overloads["test"][0]

        assert not func_overload._annotations
        assert registry["test"](0) == 1
        assert func_overload._annotations
        assert func_overload._annotations["a"] == AnnotationInfo(int, False, inspect.Parameter.POSITIONAL_OR_KEYWORD)

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

        # Tripy types should turn into class links
        @registry("test")
        def func(a: Union[int, "tripy.Tensor"]):
            """
            This func takes an int or a tensor.
            """
            pass

        print(registry["test"].__doc__)
        assert (
            registry["test"].__doc__
            == dedent(
                r"""
                *This function has multiple overloads:*

                ----------

                .. role:: sig-prename
                    :class: sig-prename descclassname
                .. role:: sig-name
                    :class: sig-name descname

                .. container:: func-overload-sig sig sig-object py

                    :sig-prename:`tripy`\ .\ :sig-name:`test`\ (a: int) -> None

                This func takes an int.

                ----------

                .. role:: sig-prename
                    :class: sig-prename descclassname
                .. role:: sig-name
                    :class: sig-name descname

                .. container:: func-overload-sig sig sig-object py

                    :sig-prename:`tripy`\ .\ :sig-name:`test`\ (a: int | :class:`tripy.Tensor`) -> None

                This func takes an int or a tensor.
                """
            ).strip()
        )

    def test_variadic_positional_args(self, registry):
        @registry("test")
        def func(*args: Any):
            return sum(args)

        assert registry["test"](1.0) == 1.0
        assert registry["test"](1.0, 2.0, 3.0) == 6.0
        assert registry["test"]() == 0

    def test_variadic_positional_and_keyword_args(self, registry):
        # ensure the interaction succeeds
        @registry("test")
        def func(a: int, *args: int, b: float, c: str):
            return a + sum(args) + int(b) + len(c)

        assert registry["test"](3, b=1.0, c="ab") == 6
        assert registry["test"](3, 4, b=1.0, c="ab") == 10
        assert registry["test"](3, 4, 5, b=1.0, c="ab") == 15

    def test_variadic_keyword_args(self, registry):
        @registry("test")
        def func(**kwargs: Dict[str, Any]):
            return sum(kwargs.values())

        assert registry["test"](a=1.0, b=2.0, c=3.0) == 6.0

    def test_sequence_check(self, registry):
        @registry("test")
        def func(int_seq: Sequence[int]) -> int:
            return sum(int_seq)

        assert registry["test"]([1, 2, 3]) == 6
        # empty should work too
        assert registry["test"]([]) == 0

    def test_sequence_no_arg_check(self, registry):
        @registry("test")
        def func(seq: Sequence) -> int:
            return len(seq)

        assert registry["test"]([1, 2, 3]) == 3
        assert registry["test"](["a", "b"]) == 2
        assert registry["test"]([]) == 0

    def test_union_check(self, registry):
        @registry("test")
        def func(n: Union[int, Sequence[int]]) -> int:
            if isinstance(n, int):
                return n
            return sum(n)

        assert registry["test"]([1, 2, 3]) == 6
        assert registry["test"](6) == 6

    def test_nested_sequence_check(self, registry):
        @registry("test")
        def func(n: Sequence[Sequence[int]]) -> int:
            if n and n[0]:
                return n[0][0]
            return -1

        assert registry["test"]([[1, 2, 3], [4, 5, 6]]) == 1

    def test_nested_union_and_sequence_check(self, registry):
        @registry("test")
        def func(n: Sequence[Union[int, Sequence[int]]]) -> int:
            if len(n) == 0:
                return 0
            if isinstance(n[0], Sequence):
                return len(n) * len(n[0])
            return len(n)

        assert registry["test"]([]) == 0
        assert registry["test"]([1, 2, 3]) == 3
        assert registry["test"]([[1, 2], [3, 4], [5, 6]]) == 6

    def test_optional_can_be_none(self, registry):
        @registry("test")
        def func(n: Optional[int]):
            return n

        assert registry["test"](None) == None
        assert registry["test"](1) == 1

    def test_error_sequence(self, registry):
        @registry("test")
        def func(n: Sequence[int]) -> int:
            return sum(n)

        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Could not find an implementation for function: 'test'\.
                Candidate overloads were:

                --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mfunc\(\)\x1b\[0m
                      \|
                  [0-9]+ \|         def func\(n: Sequence\[int\]\) \-> int:
                  [0-9]+ \|     \.\.\.
                      \|\s

                Not a valid overload because: For parameter: 'n', expected an instance of type: 'Sequence\[int\]' but got argument of type: 'List\[float\]'\.
            """
            ).strip(),
        ):
            registry["test"]([1.0, 2.0, 3.0])

    def test_error_union(self, registry):
        @registry("test")
        def func(n: Union[int, float]) -> int:
            return 0

        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Could not find an implementation for function: 'test'.
                Candidate overloads were:

                --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mfunc\(\)\x1b\[0m
                      \|
                  [0-9]+ \|         def func\(n: Union\[int, float\]\) \-> int:
                  [0-9]+ \|     \.\.\.
                      \|\s

                Not a valid overload because: For parameter: 'n', expected an instance of type: 'int | float' but got argument of type: 'List\[str\]'\.
            """
            ).strip(),
        ):
            registry["test"](["hi"])

    def test_error_inconsistent_sequence(self, registry):
        @registry("test")
        def func(n: Sequence[int]) -> int:
            return sum(n)

        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Could not find an implementation for function: 'test'.
                Candidate overloads were:

                --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mfunc\(\)\x1b\[0m
                      \|
                  [0-9]+ \|         def func\(n: Sequence\[int\]\) \-> int:
                  [0-9]+ \|     \.\.\.
                      \|\s

                Not a valid overload because: For parameter: 'n', expected an instance of type: 'Sequence\[int\]' but got argument of type: 'List\[(int \| str)|(str \| int)\]'\.
            """
            ).strip(),
        ):
            registry["test"]([1, 2, "a"])

    def test_error_not_sequence(self, registry):
        @registry("test")
        def func(n: Sequence[Sequence[int]]) -> int:
            return sum(sum(n))

        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Could not find an implementation for function: 'test'.
                Candidate overloads were:

                --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mfunc\(\)\x1b\[0m
                      \|
                  [0-9]+ \|         def func\(n: Sequence\[Sequence\[int\]\]\) \-> int:
                  [0-9]+ \|     \.\.\.
                      \|\s

                Not a valid overload because: For parameter: 'n', expected an instance of type: 'Sequence\[Sequence\[int\]\]' but got argument of type: 'List\[int\]'\.
            """
            ).strip(),
        ):
            registry["test"]([1, 2, 3])

    def test_error_nested_sequence(self, registry):
        @registry("test")
        def func(n: Sequence[Sequence[int]]) -> int:
            return sum(sum(n))

        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Could not find an implementation for function: 'test'.
                Candidate overloads were:

                --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mfunc\(\)\x1b\[0m
                      \|
                  [0-9]+ \|         def func\(n: Sequence\[Sequence\[int\]\]\) \-> int:
                  [0-9]+ \|     \.\.\.
                      \|\s

                Not a valid overload because: For parameter: 'n', expected an instance of type: 'Sequence\[Sequence\[int\]\]' but got argument of type: 'List\[List\[float\]\]'\.
            """
            ).strip(),
        ):
            registry["test"]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_error_nested_union_and_sequence(self, registry):
        @registry("test")
        def func(n: Sequence[Union[int, float]]) -> int:
            return sum(n)

        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Could not find an implementation for function: 'test'.
                Candidate overloads were:

                --> \x1b\[38;5;3m{__file__}\x1b\[0m:[0-9]+ in \x1b\[38;5;6mfunc\(\)\x1b\[0m
                      \|
                  [0-9]+ \|         def func\(n: Sequence\[Union\[int, float\]\]\) \-> int:
                  [0-9]+ \|     \.\.\.
                      \|\s

                Not a valid overload because: For parameter: 'n', expected an instance of type: 'Sequence\[int | float\]' but got argument of type: 'List\[str\]'\.
            """
            ).strip(),
        ):
            registry["test"](["a", "b", "c"])

    def test_error_variadic_positional_arg_mismatch(self, registry):
        @registry("test")
        def func(a: int, *args: int) -> int:
            return a + sum(args)

        with helper.raises(
            TripyException,
            match="Not a valid overload because: For parameter: 'args', expected an instance of type: 'int' but got argument of type: 'str'",
        ):
            registry["test"](1, 2, 3, 4, "hi")


@pytest.mark.parametrize(
    "typ, expected",
    [
        (tp.types.TensorLike, "tripy.Tensor | numbers.Number"),
        (tp.types.ShapeLike, "Sequence[int | tripy.DimensionSize]"),
        (tp.Tensor, "Tensor"),
        (torch.Tensor, "torch.Tensor"),
        (int, "int"),
        (Optional[int], "int | None"),
    ],
)
def test_str_from_type_annotation(typ, expected):
    assert str_from_type_annotation(typ) == expected


@pytest.mark.parametrize(
    "typ, expected",
    [
        (tp.Tensor([1, 2, 3]), "Tensor"),
        (torch.tensor([1, 2, 3]), "torch.Tensor"),
        (0, "int"),
        ("hi", "str"),
    ],
)
def test_type_str_from_arg(typ, expected):
    assert type_str_from_arg(typ) == expected
