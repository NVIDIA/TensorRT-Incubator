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

"""
The constraints system has two purposes:

1. Imposing input requirements.
2. Describing output guarantees.

Constraints are specified by composing one or more `Constraints` subclasses:

```py
constraint = And(
        Equal(GetDataType(GetInput("input0")), GetInput("dtype")),
        Equal(GetDataType(GetInput("input1")), GetInput("dtype")),
    )
```

We also override several bitwise operators and properties to provide a convenient shorthand.
For example, the above can be written as:

```py
constraint = (GetInput("input0").dtype == GetInput("dtype")) & (GetInput("input1").dtype == GetInput("dtype"))
```

The constraints class also provides a pattern matcher.
For example, we may want to find all constraints that check the data type of an input (`None` is a wildcard).

```py
matches = constraint.find(Equal(GetDataType(GetInput), None))
```
"""

from abc import ABC
from typing import List


class Constraints(ABC):
    """
    Base class for the entire constraints system.
    """

    def get_children(self) -> List["Constraints"]:
        children = []
        for attr_value in vars(self).values():
            if isinstance(attr_value, Constraints):
                children.append(attr_value)
            elif isinstance(attr_value, (list, tuple)):
                children.extend(v for v in attr_value if isinstance(v, Constraints))
        return children

    def find(self, pattern: "Constraints") -> List["Constraints"]:
        """
        Find all constraints in the tree that match the given pattern.

        Performs a depth-first search through the constraint tree to find all
        constraints that structurally match the given pattern, using the current
        constraint as the root node.

        Args:
            pattern: The pattern to search for (e.g., Equal(GetDataType, GetDataType)).
                    Use None as a wildcard to match anything.

        Returns:
            A list of all matching constraints found in the tree.

        Example:
            pattern = Equal(GetDataType(TensorFetcher), None)  # None matches any second argument
            matches = constraint_tree.find(pattern)
        """

        def matches_pattern(pattern: Constraints, constraint: Constraints) -> bool:
            # None is a wildcard that matches anything
            if pattern is None:
                return True

            if isinstance(pattern, type):
                return isinstance(constraint, pattern)

            if type(pattern) != type(constraint):
                return False

            # Need to index into sequences rather than comparing directly since there may be patterns in the sequence.
            if isinstance(pattern, (list, tuple)) and isinstance(constraint, (list, tuple)):
                if len(pattern) != len(constraint):
                    return False
                return all(matches_pattern(p_val, c_val) for p_val, c_val in zip(pattern, constraint))

            if not isinstance(pattern, Constraints):
                return pattern == constraint

            # Compare attributes
            pattern_attrs = vars(pattern)
            constraint_attrs = vars(constraint)

            for key, pattern_value in pattern_attrs.items():
                if key not in constraint_attrs:
                    return False

                constraint_value = constraint_attrs[key]

                if not matches_pattern(pattern_value, constraint_value):
                    return False

            return True

        matches = []

        if matches_pattern(pattern, self):
            matches.append(self)

        # Recursively search children
        for child in self.get_children():
            matches.extend(child.find(pattern))

        return matches
