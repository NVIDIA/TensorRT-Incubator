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

from dataclasses import dataclass, field
from typing import List, Optional

from nvtripy.backend.mlir import utils as mlir_utils
from nvtripy.utils.stack_info import StackInfo


@dataclass
class TraceTensor:
    """
    Represents a single tensor in the Trace
    """

    name: str
    producer: "BaseTraceOp"
    dtype: "nvtripy.common.dtype" = field(default=None, init=False)
    device: "nvtripy.common.device" = field(default=None, init=False)
    shape: List[int] = field(default=None, init=False)
    stack_info: StackInfo = field(default_factory=lambda: StackInfo([]), init=False)
    """
    Indicates the shape of the tensor. Unknown dimensions are indicated by -1.
    Generally, the shape will only be known for shape tensors.
    """

    # Whether this tensor was constructed in order to trace a computation graph for the compiler.
    is_compile_tracer: bool = False
    # Stack information for the point at which this tensor was evaluated if it was.
    # This is useful in the compiler to disallow evaluation during tracing.
    eval_stack_info: Optional[StackInfo] = field(default=None, init=False)

    def __str__(self) -> str:
        return (
            f"{self.name}: [shape=({self.shape}), "
            + (f"dtype=({self.dtype.name}), " if self.dtype is not None else "")
            + f"loc=({self.device})]"
        )

    def __repr__(self) -> str:
        # This is a hack to prevent printing the entire stack info when we print trace tensors.
        return str(self)

    def __eq__(self, other: "TraceTensor") -> bool:
        return self.name == other.name and self.stack_info == other.stack_info

    @property
    def rank(self):
        return len(self.shape)

    @rank.setter
    def rank(self, new_rank):
        self.shape = [-1] * new_rank

    def to_mlir(self):
        return mlir_utils.make_mlir_tensor(self.dtype, self.shape, self.rank)
