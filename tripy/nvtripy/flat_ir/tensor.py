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

import contextlib
import copy
from dataclasses import dataclass
from typing import Any, List, Optional

from nvtripy import utils

_BUILD_CONTEXT: List[List[Any]] = []


@dataclass(repr=False)
class FlatIRTensor:
    """
    Represents a single tensor in the FlatIR
    """

    name: str
    stack_info: utils.StackInfo
    dtype: "nvtripy.common.dtype"
    device: "nvtripy.common.device"
    rank: int
    producer: "BaseFlatIROp" = None
    shape: Optional[List[int]] = None
    reason_details: Optional[List[Any]] = None
    """
    Describes why this tensor was created.
    This is required for any intermediate tensors created by Tripy,
    but optional for tensors that can be traced back to user tensors.
    It should complete the sentence: "This operation was added in order to...".
    """
    reason_context: Optional[List[List[Any]]] = None

    @contextlib.contextmanager
    @staticmethod
    def context(ctx: List[Any]):
        try:
            global _BUILD_CONTEXT
            _BUILD_CONTEXT.append(ctx)
            yield
        finally:
            _BUILD_CONTEXT.pop()

    def to_mlir(self):
        from nvtripy.backend.mlir import utils as mlir_utils

        return mlir_utils.make_mlir_tensor(self.dtype, self.shape, self.rank)

    @staticmethod
    def build(
        dtype: "nvtripy.common.dtype",
        device: "nvtripy.common.device",
        rank: int,
        reason_details: List[Any],
        shape: List[int] = None,
    ) -> "FlatIRTensor":
        return FlatIRTensor(
            name=None,
            # Include code from the caller of this function up, and not just user code
            # since this is an intermediate tensor created within nvtripy.
            stack_info=utils.get_stack_info(include_code_index=1),
            dtype=dtype,
            device=device,
            rank=rank,
            producer=None,
            shape=shape,
            reason_details=reason_details,
            reason_context=copy.copy(_BUILD_CONTEXT),
        )

    def clone(self, reason_details, name=None) -> "FlatIRTensor":
        """
        Create a clone of the FlatIRTensor with an optional new name.
        """
        return FlatIRTensor(
            name=name,
            # Include code from the caller of this function up, and not just user code
            # since this is an intermediate tensor created within nvtripy.
            stack_info=utils.get_stack_info(include_code_index=1),
            dtype=self.dtype,
            device=self.device,
            rank=self.rank,
            producer=None,  # Reset producer for the cloned tensor
            shape=copy.deepcopy(self.shape),
            reason_details=copy.deepcopy(reason_details),
            reason_context=copy.copy(_BUILD_CONTEXT),
        )

    def __str__(self) -> str:
        return (
            f"{self.name}: [rank=({self.rank}), "
            + (f"shape=({self.shape}), " if self.shape is not None else "")
            + (f"dtype=({self.dtype.name}), " if self.dtype is not None else "")
            + f"loc=({self.device})]"
        )
