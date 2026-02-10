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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional

from nvtripy.common.datatype import DATA_TYPES
from nvtripy.frontend.constraints import AlwaysTrue, Constraints
from nvtripy.frontend.constraints.fetcher import GetDataType, GetInput
from nvtripy.frontend.constraints.logic import OneOf


@dataclass(frozen=True)
class ConstraintPass(ABC):
    name: str
    pattern: Constraints

    def predicate(self, constraint: Constraints) -> bool:
        return True

    @abstractmethod
    def rewrite(self, constraint: Constraints) -> Constraints: ...


def _optimize_children(
    constraint: Constraints,
    constraint_pass: ConstraintPass,
    pass_matches: List[Constraints],
) -> Constraints:
    """Recursively rewrite child nodes after local rewrites have been applied."""
    for attr_name, attr_value in vars(constraint).items():
        if isinstance(attr_value, Constraints):
            setattr(constraint, attr_name, _optimize_constraints(attr_value, constraint_pass, pass_matches))
            continue

        if isinstance(attr_value, (list, tuple)):
            optimized_items = []
            changed = False
            for item in attr_value:
                if isinstance(item, Constraints):
                    optimized_item = _optimize_constraints(item, constraint_pass, pass_matches)
                    optimized_items.append(optimized_item)
                    changed = changed or optimized_item is not item
                else:
                    optimized_items.append(item)
            if changed:
                new_value = tuple(optimized_items) if isinstance(attr_value, tuple) else optimized_items
                setattr(constraint, attr_name, new_value)

    return constraint


def _optimize_constraints(
    constraint: Constraints,
    constraint_pass: ConstraintPass,
    pass_matches: List[Constraints],
) -> Constraints:
    """Apply passes to this node, then recurse into its children."""
    rewritten = constraint
    if any(rewritten is match for match in pass_matches) and constraint_pass.predicate(rewritten):
        rewritten = constraint_pass.rewrite(rewritten)

    return _optimize_children(rewritten, constraint_pass, pass_matches)


class DropAllDtypesOneOf(ConstraintPass):
    def __init__(self):
        super().__init__(
            name="drop-all-dtypes-oneof",
            pattern=OneOf(GetDataType(GetInput(None)), None),
        )

    def predicate(self, constraint: Constraints) -> bool:
        return set(constraint.options).issuperset(set(DATA_TYPES.values()))

    def rewrite(self, constraint: Constraints) -> Constraints:
        return AlwaysTrue()


def _default_passes() -> Iterable[ConstraintPass]:
    return (DropAllDtypesOneOf(),)


def optimize_constraints(constraints: Optional[Constraints]) -> Optional[Constraints]:
    if constraints is None:
        return None

    passes = tuple(_default_passes())
    optimized = constraints
    for constraint_pass in passes:
        matches = optimized.find(constraint_pass.pattern)
        optimized = _optimize_constraints(optimized, constraint_pass, matches)
    return optimized
