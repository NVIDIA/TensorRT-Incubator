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
from typing import Any, List, Optional, Sequence, Tuple

from nvtripy.frontend.constraints.base import Constraints
from nvtripy.frontend.constraints.fetcher import Fetcher
from nvtripy.utils.result import Result


class Logic(Constraints):
    """
    Represents logical operations on constraints.
    """

    # When the constraint is not met, the error details should complete the sentence: "Expected ..."
    @abstractmethod
    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result: ...

    @abstractmethod
    def inverse(self) -> "Logic":
        """
        Returns the logical inverse of this constraint.
        """
        ...

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
        return self.inverse()


class OneOf(Logic):
    def __init__(self, fetcher: Fetcher, options: Sequence[Any]):
        self.fetcher = fetcher
        # Need to convert generator expressions so we can use them more than once
        self.options = list(options)

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        value = self.fetcher(args, returns)
        if value in self.options:
            return Result.ok()

        return Result.err([f"'{self.fetcher}' to be one of {self.options} (but it was '{value}')"])

    def __str__(self):
        return f"{self.fetcher} is one of {self.options}"

    def inverse(self) -> "Logic":
        return NotOneOf(self.fetcher, self.options)


class NotOneOf(Logic):
    def __init__(self, fetcher: Fetcher, options: Sequence[Any]):
        self.fetcher = fetcher
        self.options = list(options)

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        value = self.fetcher(args, returns)
        if value not in self.options:
            return Result.ok()

        return Result.err([f"'{self.fetcher}' to not be one of {self.options} (but it was '{value}')"])

    def __str__(self):
        return f"{self.fetcher} is not one of {self.options}"

    def inverse(self) -> "Logic":
        return OneOf(self.fetcher, self.options)


def get_val_or_call_fetcher(
    fetcher_or_value: Any, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None
) -> Any:
    if isinstance(fetcher_or_value, Fetcher):
        return fetcher_or_value(args, returns)
    return fetcher_or_value


class Equal(Logic):
    def __init__(self, fetcher: Fetcher, fetcher_or_value: Any):
        self.fetcher = fetcher
        self.fetcher_or_value = fetcher_or_value

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        value1 = self.fetcher(args, returns)
        value2 = get_val_or_call_fetcher(self.fetcher_or_value, args, returns)
        if value1 == value2:
            return Result.ok()

        # TODO (pranavm): If fetcher_or_value is a Fetcher, include its value in the error message.
        return Result.err([f"'{self.fetcher}' to be equal to '{self.fetcher_or_value}' (but it was '{value1}')"])

    def __str__(self):
        return f"{self.fetcher} == {self.fetcher_or_value}"

    def inverse(self) -> "Logic":
        return NotEqual(self.fetcher, self.fetcher_or_value)


class NotEqual(Logic):
    def __init__(self, fetcher: Fetcher, fetcher_or_value: Fetcher):
        self.fetcher = fetcher
        self.fetcher_or_value = fetcher_or_value

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        value1 = self.fetcher(args, returns)
        value2 = get_val_or_call_fetcher(self.fetcher_or_value, args, returns)
        if value1 != value2:
            return Result.ok()

        return Result.err([f"'{self.fetcher}' to be not equal to '{self.fetcher_or_value}' (but it was '{value1}')"])

    def __str__(self):
        return f"{self.fetcher} != {self.fetcher_or_value}"

    def inverse(self) -> "Logic":
        return Equal(self.fetcher, self.fetcher_or_value)


class And(Logic):
    def __init__(self, *constraints: Logic):
        self.constraints = constraints

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        errors = []
        for constraint in self.constraints:
            result = constraint(args, returns)
            if not result:
                errors.extend(([" and "] if errors else []) + result.error_details)
        if errors:
            return Result.err(errors)
        return Result.ok()

    def __str__(self):
        return "(" + " and ".join(str(constraint) for constraint in self.constraints) + ")"

    def inverse(self) -> "Logic":
        # De Morgan's law: not (A and B) = (not A) or (not B)
        return Or(*(constraint.inverse() for constraint in self.constraints))


class Or(Logic):
    def __init__(self, *constraints: Logic):
        self.constraints = constraints

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        all_errors = []
        for constraint in self.constraints:
            result = constraint(args, returns)
            if result:
                return Result.ok()
            all_errors.extend(([" or "] if all_errors else []) + result.error_details)
        return Result.err(all_errors)

    def __str__(self):
        return "(" + " or ".join(str(constraint) for constraint in self.constraints) + ")"

    def inverse(self) -> "Logic":
        # De Morgan's law: not (A or B) = (not A) and (not B)
        return And(*(constraint.inverse() for constraint in self.constraints))
