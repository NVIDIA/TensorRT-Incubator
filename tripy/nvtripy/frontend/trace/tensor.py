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

from nvtripy import utils
from typing import Optional, List


@dataclass
class TraceTensor:
    """
    Represents a single tensor in the Trace
    """

    name: str
    stack_info: utils.StackInfo
    dtype: "nvtripy.common.dtype"
    device: "nvtripy.common.device"
    producer: "BaseTraceOp"
    shape: List[int]
    """
    Indicates the shape of the tensor. Unknown dimensions are indicated by -1.
    Generally, the shape will only be known for shape tensors.
    """

    # Whether this tensor was constructed in order to trace a computation graph for the compiler.
    is_compile_tracer: bool = False
    # Stack information for the point at which this tensor was evaluated if it was.
    # This is useful in the compiler to disallow evaluation during tracing.
    eval_stack_info: Optional[utils.StackInfo] = None

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

    def to_flat_ir(self) -> "FlatIRTensor":
        from nvtripy.flat_ir.tensor import FlatIRTensor

        tensor = FlatIRTensor(
            name=self.name,
            stack_info=self.stack_info,
            dtype=self.dtype,
            device=self.device,
            rank=self.rank,
            # Only set shape if known:
            shape=self.shape if -1 not in self.shape else None,
        )
        return tensor

    @property
    def rank(self):
        return len(self.shape)

    @rank.setter
    def rank(self, new_rank):
        self.shape = [-1] * new_rank
