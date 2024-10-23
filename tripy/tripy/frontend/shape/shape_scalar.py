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

from typing import Any, Optional, Sequence, Union

import tripy.frontend.utils as frontend_utils
from tripy import constraints, export, utils
from tripy.common.datatype import int32
from tripy.common.exception import raise_error
from tripy.frontend.tensor import Tensor
from tripy.utils.stack_info import StackInfo
from textwrap import indent
from typing import Any, Optional

import mlir_tensorrt.runtime.api as runtime

# Import ops to populate the registry before we define our Tensor class
import tripy.frontend.ops
import tripy.frontend.trace.ops
from tripy import export, utils
from tripy.backend.mlir import memref
from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops import Storage
from tripy.frontend.trace.tensor import TraceTensor
from tripy.utils.stack_info import StackInfo


# TODO (pranavm): Make ShapeScalar not a subclass of tensor.
@export.public_api(document_under="shape")
class ShapeScalar:
    """
    Scalar shape is a tensor used to represent a scalar value extracted from a shape tensor.
    ShapeScalars are scalars (rank 0) of non-negative integer (using int32 as the datatype).
    """

    def __init__(
        self,
        data: Any,
        name: Optional[str] = None,
    ) -> None:
        r"""
        Args:
            data: The value of the ShapeScalar, which should be a scalar integer.
            name: An optional name
        """

        from tripy.common.exception import raise_error

        if isinstance(data, Tensor):
            # these fields can be None in the case of an uninitialized tensor (like Tensor(None))
            if data.trace_tensor.rank is not None and data.trace_tensor.rank != 0:
                raise_error(
                    f"Scalar shape tensors must be of rank 0, but input tensor is rank {data.rank}", details=[data]
                )
            if data.dtype is not None and data.dtype != int32:
                raise_error(
                    f"Scalar shape tensor must have int32 member, but input tensor has data type {data.dtype}",
                    details=[data],
                )

            # the shape of data should correspond to the given rank
            super().__init__(data=None, dtype=int32, name=name, device=data.device)
            # share the underlying data
            self.trace_tensor = data.trace_tensor
            self.stack_info = data.stack_info
        else:
            shape = data.shape if hasattr(data, "shape") else utils.get_shape(data)
            device = data.device if hasattr(data, "device") else None
            if len(shape) != 0:
                raise_error(
                    f"Tensors used to represent scalar shapes must be of rank 0, but given shape {shape} has rank {len(shape)}."
                )
            super().__init__(data=data, dtype=int32, name=name, device=device)

    def __int__(self) -> int:
        return self.tolist()

    def __repr__(self) -> str:
        # denote the representation as a shape rather than a tensor
        tensor_repr = super().__repr__()
        assert tensor_repr[:6] == "tensor"
        return "shape_scalar" + tensor_repr[6:]

    def __str__(self) -> str:
        val = self.tolist()
        assert isinstance(val, int)
        return f"shape_scalar({val})"
