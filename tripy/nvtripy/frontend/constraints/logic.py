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
from abc import abstractmethod
from typing import Any, List, Sequence, Tuple

from nvtripy.frontend.constraints.base import Constraints
from nvtripy.frontend.constraints.fetcher import Fetcher
from nvtripy.utils.result import Result

# TODO (pranavm): Maybe error details should say what went wrong, not what was expected?
# TODO (pranavm): Maybe return a list of violated constraints? Then we can construct error messages outside.


class Logic(Constraints):
    """
    Represents logical operations on constraints.
    """

    @abstractmethod
    def __call__(self, args: List[Tuple[str, Any]]) -> Result: ...

    def __and__(self, other: "Logic") -> "Logic":
        if isinstance(self, And):
            return And(*self.constraints, other)
        elif isinstance(other, And):
            return And(self, *other.constraints)
        return And(self, other)

    def __or__(self, other: "Logic") -> "Logic":
        if isinstance(self, Or):
            return Or(*self.constraints, other)
        elif isinstance(other, Or):
            return Or(self, *other.constraints)
        return Or(self, other)

    def __invert__(self) -> "Logic":
        if isinstance(self, Equal):
            return NotEqual(self.fetcher, self.fetcher_or_value)
        return Not(self)


class OneOf(Logic):
    def __init__(self, fetcher: Fetcher, options: Sequence[Any]):
        self.fetcher = fetcher
        # Need to convert generator expressions so we can use them more than once
        self.options = list(options)

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        value = self.fetcher(args)
        if value in self.options:
            return Result.ok()

        return Result.err([f"'{self.fetcher}' to be one of {self.options}, but got {value}"])

    def __str__(self):
        return f"{self.fetcher} is one of {self.options}"


def get_val_or_call_fetcher(fetcher_or_value: Any, args: List[Tuple[str, Any]]) -> Any:
    if isinstance(fetcher_or_value, Fetcher):
        return fetcher_or_value(args)
    return fetcher_or_value


class Equal(Logic):
    def __init__(self, fetcher: Fetcher, fetcher_or_value: Any):
        self.fetcher = fetcher
        self.fetcher_or_value = fetcher_or_value

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        value1 = self.fetcher(args)
        value2 = get_val_or_call_fetcher(self.fetcher_or_value, args)
        if value1 == value2:
            return Result.ok()

        return Result.err([f"'{self.fetcher}' to be equal to '{self.fetcher_or_value}'"])

    def __str__(self):
        return f"{self.fetcher} == {self.fetcher_or_value}"


class NotEqual(Logic):
    def __init__(self, fetcher: Fetcher, fetcher_or_value: Fetcher):
        self.fetcher = fetcher
        self.fetcher_or_value = fetcher_or_value

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        value1 = self.fetcher(args)
        value2 = get_val_or_call_fetcher(self.fetcher_or_value, args)
        if value1 != value2:
            return Result.ok()

        return Result.err([f"'{self.fetcher}' to be not equal to '{self.fetcher_or_value}'"])

    def __str__(self):
        return f"{self.fetcher} != {self.fetcher_or_value}"


# TODO (pranavm): Make And and Or combine errors and include a prefix like: "{...} but condition XYZ failed."
class And(Logic):
    def __init__(self, *constraints: Logic):
        self.constraints = constraints

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        errors = []
        for constraint in self.constraints:
            result = constraint(args)
            if not result:
                errors.extend(([" and "] if errors else []) + result.error_details)
        if errors:
            return Result.err(errors)
        return Result.ok()

    def __str__(self):
        return "(" + " and ".join(str(constraint) for constraint in self.constraints) + ")"


class Or(Logic):
    def __init__(self, *constraints: Logic):
        self.constraints = constraints

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        all_errors = []
        for constraint in self.constraints:
            result = constraint(args)
            if result:
                return Result.ok()
            all_errors.extend(([" or "] if all_errors else []) + result.error_details)
        return Result.err(all_errors)

    def __str__(self):
        return "(" + " or ".join(str(constraint) for constraint in self.constraints) + ")"


class Not(Logic):
    def __init__(self, constraint: Logic):
        self.constraint = constraint

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        result = self.constraint(args)
        if result:
            return Result.err([str(self)])
        return Result.ok()

    def __str__(self):
        return f"not {self.constraint}"
