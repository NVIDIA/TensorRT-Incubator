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
from typing import Optional

import nvtripy.frontend.trace.ops.utils as op_utils
import nvtripy.frontend.utils as frontend_utils
from nvtripy import export, utils, wrappers
from nvtripy.common import datatype
from nvtripy.frontend.trace.ops import utils as op_utils
from nvtripy.frontend.trace.ops.base import BaseTraceOp
from nvtripy.types import ShapeLike, TensorLike


@dataclass(repr=False)
class Fill(BaseTraceOp):
    dtype: datatype.dtype

    infer_rank = op_utils.InferRankPolicies.same_as_shape_of_shape_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from nvtripy.common import device

        self.outputs[0].device = device.create_directly("gpu", 0)

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import ConvertOp, DynamicBroadcastOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        const_val_tensor = None
        assert (
            len(inputs) == 2
        ), f"Expected value of Fill to be provided as input. Expected 2 inputs, got {len(inputs)}."
        const_val_tensor = inputs[1]
        if inputs[1].dtype != outputs[0].dtype:
            out = FlatIRTensor.build(
                shape=(),
                rank=0,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=[f"create the constant value tensor for a fill operation"],
            )

            ConvertOp.build([const_val_tensor], [out])
            const_val_tensor = out
        DynamicBroadcastOp.build(
            [const_val_tensor, inputs[0]],
            outputs,
            broadcast_dim=[],
        )


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"dtype": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    convert_to_tensors=True,
)
def full(shape: ShapeLike, value: TensorLike, dtype: "nvtripy.dtype" = datatype.float32) -> "nvtripy.Tensor":
    """
    Returns a tensor of the desired shape with all values set to the specified value.

    Args:
        shape: The desired shape.
        value: A scalar value to fill the resulting tensor.
        dtype: The desired data type.

    Returns:
        A tensor of shape ``shape``.

    .. code-block:: python
        :linenos:
        :caption: Example

        output = tp.full(shape=[2, 3], value=2)

        assert np.array_equal(cp.from_dlpack(output).get(), np.full([2, 3], 2, dtype=np.float32))
    """
    return Fill.build([shape, value], dtype=dtype)


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"input": "T1", "dtype": "T2", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    convert_to_tensors=True,
)
def full_like(input: "nvtripy.Tensor", value: TensorLike, dtype: Optional["nvtripy.dtype"] = None) -> "nvtripy.Tensor":
    """
    Returns a tensor of the same shape and data type as the input tensor, with all values
    set to the specified value.

    Args:
        input: Input tensor.
        value: A scalar value to fill the resulting tensor.
        dtype: The desired data type. This will override the data type inferred from the input tensor.

    Returns:
        A tensor of the same shape and data type (unless ``dtype`` is provided) as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([[1, 2], [3, 4]])
        output = tp.full_like(input, value=2)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[2, 2], [2, 2]], dtype=np.float32))
    """
    return Fill.build(
        [frontend_utils.tensor_from_shape_like(input.shape), value], dtype=utils.default(dtype, input.dtype)
    )
