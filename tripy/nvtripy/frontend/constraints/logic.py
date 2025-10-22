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

    def __invert__(self) -> "Logic":
        if isinstance(self, Equal):
            return NotEqual(self.fetcher1, self.fetcher2)
        return Not(self)


class OneOf(Logic):
    def __init__(self, fetcher: Fetcher, options: Sequence[Any]):
        self.fetcher = fetcher
        self.options = options

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        value = self.fetcher(args)
        if value in self.options:
            return Result.ok()

        return Result.err([f"Expected {self.fetcher} to be one of {self.options}, but got {value}."])

    def __str__(self):
        return f"{self.fetcher} is one of {self.options}"


class Equal(Logic):
    def __init__(self, fetcher1: Fetcher, fetcher2: Fetcher):
        self.fetcher1 = fetcher1
        self.fetcher2 = fetcher2

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        value1 = self.fetcher1(args)
        value2 = self.fetcher2(args)
        if value1 == value2:
            return Result.ok()

        return Result.err([f"Expected {self.fetcher1} to be equal to {self.fetcher2}, but got {value1} and {value2}."])

    def __str__(self):
        return f"{self.fetcher1} == {self.fetcher2}"


class NotEqual(Logic):
    def __init__(self, fetcher1: Fetcher, fetcher2: Fetcher):
        self.fetcher1 = fetcher1
        self.fetcher2 = fetcher2

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        value1 = self.fetcher1(args)
        value2 = self.fetcher2(args)
        if value1 != value2:
            return Result.ok()

        return Result.err([f"Expected {self.fetcher1} to be not equal to {self.fetcher2}, but both were {value1}."])

    def __str__(self):
        return f"{self.fetcher1} != {self.fetcher2}"


class And(Logic):
    def __init__(self, *constraints: Logic):
        self.constraints = constraints

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        errors = []
        for constraint in self.constraints:
            result = constraint(args)
            if not result:
                errors.extend(result.error_details)
        if errors:
            return Result.err(errors)
        return Result.ok()

    def __str__(self):
        return " and ".join(str(constraint) for constraint in self.constraints)


class Not(Logic):
    def __init__(self, constraint: Logic):
        self.constraint = constraint

    def __call__(self, args: List[Tuple[str, Any]]) -> Result:
        result = self.constraint(args)
        if result:
            return Result.err([f"Expected NOT {self.constraint}, but it was satisfied."])
        return Result.ok()

    def __str__(self):
        return f"NOT ({self.constraint})"
