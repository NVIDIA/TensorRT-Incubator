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

import numbers
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import tripy.frontend.trace.ops.utils as op_utils
import tripy.frontend.utils as frontend_utils
from tripy import export, utils, constraints
from tripy.common import datatype
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.common.datatype import DATA_TYPES


@dataclass(repr=False)
class Fill(BaseTraceOp):
    value: float
    output_rank: int
    dtype: datatype.dtype

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def infer_rank(self):
        if self.output_rank is None:
            input_shape = op_utils.get_trace_shape(self.inputs[0])
            assert len(input_shape) == 1, f"Expected rank of shape tensor to be 1, got {len(input_shape)}"
            assert (
                input_shape[0] >= 0
            ), f"Incorrect shape of shape tensor, expected shape to be positive, got {input_shape[0]}"
            self.output_rank = input_shape[0]
        self.outputs[0].rank = self.output_rank

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConstantOp, ConvertOp, DynamicBroadcastOp
        from tripy.flat_ir.tensor import FlatIRTensor

        const_val_tensor = None
        if self.value is not None:
            const_val_tensor = FlatIRTensor.build(
                shape=(),
                rank=0,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=[f"create the constant value tensor (containing {self.value}) for a fill operation"],
            )
            ConstantOp.build([], [const_val_tensor], data=self.value)
        else:
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


@frontend_utils.convert_shape_inputs(["shape"])
def full_impl(
    shape: "tripy.types.ShapeLike",
    value: Union[numbers.Number, "tripy.Tensor"],
    dtype: "tripy.dtype",
    output_rank: int,
) -> "tripy.Tensor":
    from tripy.frontend.tensor import Tensor

    if isinstance(value, Tensor):
        return Fill.build([shape, value], None, output_rank, dtype)

    return Fill.build([shape], value, output_rank, dtype)


@export.public_api(document_under="operations/initializers")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    dtype_constraints={"dtype": "T1", constraints.RETURN_VALUE: "T1"},
)
def full(
    shape: "tripy.types.ShapeLike",
    value: Union[numbers.Number, "tripy.Tensor"],
    dtype: "tripy.dtype" = datatype.float32,
) -> "tripy.Tensor":
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
    output_rank = len(shape) if isinstance(shape, Sequence) else None
    return full_impl(shape, value, dtype, output_rank)


@export.public_api(document_under="operations/initializers")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    dtype_constraints={"input": "T1", "dtype": "T2", constraints.RETURN_VALUE: "T2"},
)
def full_like(input: "tripy.Tensor", value: numbers.Number, dtype: Optional["tripy.dtype"] = None) -> "tripy.Tensor":
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
    return full_impl(input.shape, value, utils.default(dtype, input.dtype), input.rank)
