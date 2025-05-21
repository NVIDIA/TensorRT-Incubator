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

from dataclasses import dataclass, field
from typing import Tuple, Optional

from nvtripy import constants
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
    # Indicates the shape of the tensor. Unknown dimensions are indicated by DYNAMIC_DIM.
    # Generally, the shape will only be known for shape tensors.
    shape: Tuple[int] = field(default=None, init=False)
    stack_info: StackInfo = field(default=None, init=False)

    # Whether this tensor was constructed in order to trace a computation graph for the compiler.
    is_compile_tracer: bool = False
    # Stack information for the point at which this tensor was evaluated if it was.
    # This is useful in the compiler to disallow evaluation during tracing.
    eval_stack_info: Optional[StackInfo] = field(default=None, init=False)

    frontend_tensor: "nvtripy.Tensor" = field(default=None, init=False)

    def type_descriptor(self) -> str:
        type_elements = [str(s) if s != constants.DYNAMIC_DIM else "?" for s in self.shape]
        type_elements.append(self.dtype.shortname if self.dtype is not None else "?")
        return f"<{'x'.join(type_elements)}:{self.device}>"

    def __str__(self) -> str:
        return f"{self.name} : tensor{self.type_descriptor()}"

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
        self.shape = (constants.DYNAMIC_DIM,) * new_rank

    def to_mlir(self):
        return mlir_utils.make_mlir_tensor(self.dtype, self.shape, self.rank)
