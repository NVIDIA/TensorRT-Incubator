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
from abc import abstractmethod
from textwrap import indent
from typing import Any, List, Optional, Sequence, Tuple

from nvtripy.frontend.constraints.base import Constraints
from nvtripy.frontend.constraints.doc_str import doc_str
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


class AlwaysTrue(Logic):
    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        return Result.ok()

    def __str__(self):
        return "true"

    def doc_str(self) -> str:
        return "true"

    def inverse(self) -> "Logic":
        return AlwaysFalse()


class AlwaysFalse(Logic):
    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        return Result.err(["true"])

    def __str__(self):
        return "false"

    def doc_str(self) -> str:
        return "false"

    def inverse(self) -> "Logic":
        return AlwaysTrue()


class OneOf(Logic):
    def __init__(self, fetcher: Fetcher, options: Optional[Sequence[Any]]):
        super().__init__()
        self.fetcher = fetcher
        # Need to convert generator expressions so we can use them more than once.
        # `None` is allowed to support pattern matching wildcards.
        self.options = list(options) if options is not None else None

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        if self.options is None:
            raise_error("OneOf constraint cannot be evaluated with wildcard options.")
        value = self.fetcher(args, returns)
        if value in self.options:
            return Result.ok()

        return Result.err([f"'{self.fetcher}' to be one of {self.options} (but it was '{value}')"])

    def __str__(self):
        return f"{self.fetcher} is one of {self.options}"

    def doc_str(self) -> str:
        if self.options is None:
            return f"{doc_str(self.fetcher)} is one of [*]"
        return f"{doc_str(self.fetcher)} is one of [{', '.join(f'{doc_str(opt)}' for opt in self.options)}]"

    def inverse(self) -> "Logic":
        return NotOneOf(self.fetcher, self.options)


class NotOneOf(Logic):
    def __init__(self, fetcher: Fetcher, options: Sequence[Any]):
        super().__init__()
        self.fetcher = fetcher
        self.options = list(options)

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        value = self.fetcher(args, returns)
        if value not in self.options:
            return Result.ok()

        return Result.err([f"'{self.fetcher}' to not be one of {self.options} (but it was '{value}')"])

    def __str__(self):
        return f"{self.fetcher} is not one of {self.options}"

    def doc_str(self) -> str:
        return f"{doc_str(self.fetcher)} is not one of [{', '.join(f'{doc_str(opt)}' for opt in self.options)}]"

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
        super().__init__()
        self.fetcher = fetcher
        self.fetcher_or_value = fetcher_or_value

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        value1 = self.fetcher(args, returns)
        value2 = get_val_or_call_fetcher(self.fetcher_or_value, args, returns)

        # Avoid triggering overloaded equality implementations (e.g., on Tensor) when comparing to None.
        if value1 is None or value2 is None:
            if value1 is value2:
                return Result.ok()
        elif value1 == value2:
            return Result.ok()

        # TODO (pranavm): If fetcher_or_value is a Fetcher, include its value in the error message.
        return Result.err([f"'{self.fetcher}' to be equal to '{self.fetcher_or_value}' (but it was '{value1}')"])

    def __str__(self):
        return f"{self.fetcher} == {self.fetcher_or_value}"

    def doc_str(self) -> str:
        return f"{doc_str(self.fetcher)} == {doc_str(self.fetcher_or_value)}"

    def inverse(self) -> "Logic":
        return NotEqual(self.fetcher, self.fetcher_or_value)


class NotEqual(Logic):
    def __init__(self, fetcher: Fetcher, fetcher_or_value: Fetcher):
        super().__init__()
        self.fetcher = fetcher
        self.fetcher_or_value = fetcher_or_value

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        value1 = self.fetcher(args, returns)
        value2 = get_val_or_call_fetcher(self.fetcher_or_value, args, returns)

        # Avoid triggering overloaded inequality implementations (e.g., on Tensor) when comparing to None.
        if value1 is None or value2 is None:
            if value1 is not value2:
                return Result.ok()
        elif value1 != value2:
            return Result.ok()

        return Result.err([f"'{self.fetcher}' to be not equal to '{self.fetcher_or_value}' (but it was '{value1}')"])

    def __str__(self):
        return f"{self.fetcher} != {self.fetcher_or_value}"

    def doc_str(self) -> str:
        return f"{doc_str(self.fetcher)} != {doc_str(self.fetcher_or_value)}"

    def inverse(self) -> "Logic":
        return Equal(self.fetcher, self.fetcher_or_value)


class And(Logic):
    def __init__(self, *constraints: Logic):
        super().__init__()
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

    def doc_str(self) -> str:
        return ", **and**\n".join("- " + indent(doc_str(constraint), "  ").lstrip() for constraint in self.constraints)

    def inverse(self) -> "Logic":
        # De Morgan's law: not (A and B) = (not A) or (not B)
        return Or(*(constraint.inverse() for constraint in self.constraints))


class Or(Logic):
    def __init__(self, *constraints: Logic):
        super().__init__()
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

    def doc_str(self) -> str:
        return "(" + " *or* ".join(doc_str(constraint) for constraint in self.constraints) + ")"

    def inverse(self) -> "Logic":
        # De Morgan's law: not (A or B) = (not A) and (not B)
        return And(*(constraint.inverse() for constraint in self.constraints))


class If(Logic):
    def __init__(self, condition: Logic, then_branch: Logic, else_branch: Optional[Logic] = None):
        super().__init__()
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Result:
        condition_result = self.condition(args, returns)
        if condition_result:
            return self.then_branch(args, returns)
        else:
            return self.else_branch(args, returns) if self.else_branch else Result.ok()

    def __str__(self):
        if self.else_branch:
            return f"if ({self.condition}) then ({self.then_branch}) else ({self.else_branch})"
        return f"if ({self.condition}) then ({self.then_branch})"

    def doc_str(self) -> str:
        if self.else_branch:
            return f"{doc_str(self.then_branch)} **if** {doc_str(self.condition)}, **otherwise** {doc_str(self.else_branch)}"
        return f"if {doc_str(self.condition)}, then {doc_str(self.then_branch)}"

    def inverse(self) -> "Logic":
        return If(self.condition, self.then_branch.inverse(), self.else_branch.inverse() if self.else_branch else None)
