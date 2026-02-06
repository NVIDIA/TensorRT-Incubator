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
import numbers
from typing import Any, List, Optional, Sequence, Tuple

from nvtripy.common import datatype
from nvtripy.common.datatype import dtype as tp_dtype
from nvtripy.common.exception import raise_error
from nvtripy.frontend.constraints.base import Constraints


class Fetcher(Constraints):
    """
    Fetches a value based on the function parameters or return value.
    """

    @abstractmethod
    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Any: ...

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
        super().__init__()
        self.name = name

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Any:
        for name, value in args:
            if name == self.name:
                return value
        assert False, f"Input '{self.name}' not found in arguments."

    def __str__(self):
        return self.name

    def doc_str(self) -> str:
        return f"``{self.name}``"


class GetReturn(ValueFetcher):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Any:
        assert returns is not None, "No return values available."
        return returns[self.index]

    def __str__(self):
        return f"return[{self.index}]"

    def doc_str(self) -> str:
        return f"``return[{self.index}]``"


class GetDataType(Fetcher):
    def __init__(self, value_fetcher: ValueFetcher):
        super().__init__()
        self.value_fetcher = value_fetcher

    def __call__(self, args: List[Tuple[str, Any]], returns: Optional[Tuple[Any]] = None) -> Any:
        from nvtripy.frontend.tensor import Tensor

        def get_arg_dtype(arg: Any) -> tp_dtype:
            if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
                arg_dtypes = [get_arg_dtype(elem) for elem in arg]

                if len(arg_dtypes) == 0:
                    raise_error(
                        f"Could not determine data type of {self.value_fetcher}",
                        [
                            "Empty sequence argument.\n",
                            f"For parameter: '{self.value_fetcher}', the sequence must contain at least one element.",
                        ],
                    )

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
            elif isinstance(arg, tp_dtype):
                arg_dtype = arg
            elif isinstance(arg, bool):
                arg_dtype = datatype.bool
            elif isinstance(arg, numbers.Integral):
                arg_dtype = datatype.int32 if datatype.INT32_MIN <= arg <= datatype.INT32_MAX else datatype.int64
            elif isinstance(arg, numbers.Real):
                arg_dtype = datatype.float32
            else:
                raise_error(
                    f"Could not determine data type of {self.value_fetcher}",
                    [f"Expected a tensor or data type argument for {self.value_fetcher}, but got: {arg}"],
                )
            return arg_dtype

        tensor = self.value_fetcher(args, returns)
        return get_arg_dtype(tensor)

    def __str__(self):
        return f"{self.value_fetcher}.dtype"

    def doc_str(self) -> str:
        # Intentionally do not use doc_str() on the value_fetcher so we can wrap it in backticks correctly.
        return f"``{self.value_fetcher}.dtype``"
