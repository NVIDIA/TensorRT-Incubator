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

from dataclasses import dataclass
from typing import Any, Optional, List
from nvtripy.utils.types import str_from_type_annotation


@dataclass
class Result:
    """
    Represents the returned value of a function or an error message.
    This is conceptually similar to Rust's `std::result`.
    """

    value: Optional[Any]
    error_details: Optional[List[str]]
    is_ok: bool

    @staticmethod
    def ok(value: Any = None) -> "Result":
        return Result(value, None, is_ok=True)

    @staticmethod
    def err(error_details: List[str]) -> "Result":
        return Result(None, error_details, is_ok=False)

    def __bool__(self) -> bool:
        return self.is_ok

    def __getattribute__(self, name: str) -> Any:
        if name == "value":
            assert self.is_ok, "Cannot retrieve value of an error result"
        if name == "error_details":
            assert not self.is_ok, "Cannot retrieve error details of an ok result"

        return super().__getattribute__(name)

    def __class_getitem__(cls, item):
        return f"{cls.__name__}[{str_from_type_annotation(item)}]"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        if self.is_ok:
            return f"Result.ok({self.value})"
        return f"Result.err({self.error_details})"
