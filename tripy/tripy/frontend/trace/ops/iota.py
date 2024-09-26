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
from typing import Optional, Sequence, Union

from tripy import export, utils, constraints
from tripy.common import datatype
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.common.exception import raise_error


@dataclass(repr=False)
class Iota(BaseTraceOp):
    dim: int
    output_rank: int
    dtype: datatype.dtype

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_rank(self):
        if self.output_rank is None:
            if self.inputs[0].shape is None:
                from tripy.backend.mlir.utils import ShapeContext

                out_shape = ShapeContext().get_shape_of_dynamic_trace_tensor(self.inputs[0])
                assert len(out_shape) == 1
                assert out_shape[0] >= 0, f"incorrect shape computation {out_shape}"
                self.output_rank = out_shape[0]
            else:
                self.output_rank = self.inputs[0].shape[0]

        # Iota requires inputs[0] to be statically shaped
        if self.inputs[0].shape is None:
            self.inputs[0].shape = (self.output_rank,)

        if self.dim < 0:
            self.dim += self.output_rank
        self.outputs[0].rank = self.output_rank

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicIotaOp

        DynamicIotaOp.build(inputs, outputs, dim=self.dim)


@frontend_utils.convert_shape_inputs(["shape"])
def iota_impl(
    shape: Union["tripy.Shape", Sequence[int]], dim: int, dtype: datatype.dtype, output_rank: int
) -> "tripy.Tensor":
    from tripy.frontend.trace.ops.cast import cast

    # Allocate a float32 tensor and cast the output to dtype. `tensorrt.linspace` op result #0 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor of 32-bit float or 32-bit signless integer values.
    if dtype not in (datatype.float32, datatype.int32, datatype.int64):
        result = Iota.build([shape], dim, output_rank, datatype.float32)
        return cast(result, dtype)

    return Iota.build([shape], dim, output_rank, dtype)


@export.public_api(document_under="operations/initializers")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "bool"],
    },
    dtype_constraints={"dtype": "T1", constraints.RETURN_VALUE: "T1"},
)
def iota(
    shape: "tripy.types.ShapeLike",
    dim: int = 0,
    dtype: datatype.dtype = datatype.float32,
) -> "tripy.Tensor":
    """
    Fills an output tensor with consecutive values starting from zero along the given dimension.

    Args:
        shape: The desired shape.
        dim: Dimension along which to perform the iota operation.
            This cannot exceed the rank of the specified shape.
        dtype: The desired data type.

    Returns:
        A tensor of shape ``shape`` and data type ``dtype``.

    .. code-block:: python
        :linenos:
        :caption: Example

        output = tp.iota((3,), dim=-1)

        assert np.array_equal(cp.from_dlpack(output).get(), np.arange(0, 3, dtype=np.float32))
    """
    output_rank = len(shape) if isinstance(shape, Sequence) else None
    return iota_impl(shape, dim, dtype, output_rank)


@export.public_api(document_under="operations/initializers")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "bool"],
    },
    dtype_constraints={"input": "T1", "dtype": "T2", constraints.RETURN_VALUE: "T2"},
)
def iota_like(input: "tripy.Tensor", dim: int = 0, dtype: Optional[datatype.dtype] = None) -> "tripy.Tensor":
    """
    Returns a tensor of the same shape and data type as the input tensor, with consecutive values
    starting from zero along the given dimension.

    Args:
        input: Input tensor.
        dim: Dimension along which to perform the iota operation.
            This cannot exceed the rank of the specified shape.
        dtype: The desired data type. This will override the data type inferred from the input tensor.

    Returns:
        A tensor of the same shape and data type (unless ``dtype`` is provided) as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1, 2, 3])
        output = tp.iota_like(input)

        assert np.array_equal(cp.from_dlpack(output).get(), np.arange(0, 3, dtype=np.float32))
    """
    return iota_impl(input.shape, dim, utils.default(dtype, input.dtype), input.rank)
