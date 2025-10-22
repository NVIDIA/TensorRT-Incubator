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

from nvtripy.common.datatype import dtype as tp_dtype
from nvtripy.common.exception import raise_error
from nvtripy.frontend.constraints.base import Constraints


class Fetcher(Constraints):
    """
    Fetches a value based on the function parameters or return value.
    """

    @abstractmethod
    def __call__(self, args: List[Tuple[str, Any]]) -> Any: ...

    def __eq__(self, other: "Fetcher") -> "Equal":
        from nvtripy.frontend.constraints.logic import Equal

        return Equal(self, other)

    def __ne__(self, other: "Fetcher") -> "NotEqual":
        from nvtripy.frontend.constraints.logic import NotEqual

        return NotEqual(self, other)


class ValueFetcher(Fetcher):
    @property
    def dtype(self) -> "GetDataType":
        return GetDataType(self)


class GetInput(ValueFetcher):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, args: List[Tuple[str, Any]]) -> Any:
        for name, value in args:
            if name == self.name:
                return value
        assert False, f"Input '{self.name}' not found in arguments."

    def __str__(self):
        return self.name


class GetReturn(ValueFetcher):
    def __init__(self, index: int):
        self.index = index

    def __call__(self, args: List[Tuple[str, Any]]) -> Any:
        raise NotImplementedError(
            "GetReturn is only used to describe output guarantees and must not be called for input validation purposes."
        )

    def __str__(self):
        return f"return[{self.index}]"


class GetDataType(Fetcher):
    def __init__(self, value_fetcher: ValueFetcher):
        self.value_fetcher = value_fetcher

    def __call__(self, args: List[Tuple[str, Any]]) -> Any:
        from nvtripy.frontend.tensor import Tensor

        def get_arg_dtype(arg: Any) -> tp_dtype:
            if isinstance(arg, Sequence):
                arg_dtypes = [get_arg_dtype(elem) for elem in arg]

                if len(set(arg_dtypes)) != 1:
                    raise_error(
                        f"Could not determine data type of {self.value_fetcher}",
                        [
                            f"Mismatched data types in sequence argument.\n",
                            f"For parameter: '{self.value_fetcher}', all arguments must have the same data type, but got: "
                            f"{arg_dtypes}",
                        ],
                    )
                arg_dtype = arg_dtypes[0]
            elif isinstance(arg, Tensor):
                arg_dtype = arg.dtype
            else:
                raise_error(
                    f"Could not determine data type of {self.value_fetcher}",
                    [f"Expected a tensor or data type argument for {self.value_fetcher}, but got: {arg}"],
                )
            return arg_dtype

        tensor = self.value_fetcher(args)
        return get_arg_dtype(tensor)

    def __str__(self):
        return f"{self.value_fetcher}.dtype"
