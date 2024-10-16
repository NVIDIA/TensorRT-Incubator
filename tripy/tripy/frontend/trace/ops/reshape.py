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
from typing import Optional, Sequence, Tuple, List, Union
from tripy import export, utils, constraints
from tripy.common.exception import raise_error
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Reshape(BaseTraceOp):

    output_rank: int
    output_len: Optional[int] = None  # only used to help with infer_len for a shape input

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_len(self):
        if self.output_len is not None:
            return [self.output_len]
        # skip inference for now because it requires obtaining the concrete _value_ of the second input,
        # not just its shape
        return [None]

    def infer_tensor_variants(self, inputs):
        from tripy.frontend.shape import Shape
        from tripy.utils import Result

        # Only wrap the reshaped output if the result is rank 1
        if isinstance(inputs[0], Shape) and self.output_rank == 1:
            return Result.ok([Shape])
        return Result.ok([None])

    def infer_rank(self):
        if self.output_rank is None:
            shape_of_shape_input = op_utils.get_trace_shape(self.inputs[1])
            assert len(shape_of_shape_input) == 1
            assert shape_of_shape_input[0] >= 0, f"incorrect shape computation {shape_of_shape_input}"
            self.outputs[0].rank = shape_of_shape_input[0]
        else:
            self.outputs[0].rank = self.output_rank

    @frontend_utils.make_function
    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicReshapeOp

        DynamicReshapeOp.build(inputs, outputs)


@frontend_utils.convert_shape_inputs(["shape"])
def reshape_impl(
    input: "tripy.Tensor", shape: Sequence, output_rank: int, output_len: Optional[int] = None
) -> "tripy.Tensor":
    return Reshape.build([input, shape], output_rank, output_len)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def reshape(input: "tripy.Tensor", shape: "tripy.types.ShapeLike") -> "tripy.Tensor":
    """
    Returns a new tensor with the contents of the input tensor in the specified shape.

    Args:
        input: The input tensor.
        shape: The desired compatible shape. If a shape dimension is -1, its value
            is inferred based on the other dimensions and the number of elements in the input.
            Atmost one dimension can be -1.

    Returns:
        A new tensor with the specified shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.iota((2, 3), dtype=tp.float32)
        output = tp.reshape(input, (1, 6))

        assert np.array_equal(cp.from_dlpack(output).get(), np.reshape(cp.from_dlpack(input).get(), (1, 6)))
    """
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.shape import Shape

    if isinstance(shape, Tensor):
        return Reshape.build([input, shape], None)

    def compute_unknown_dim(input_tensor, out_shape):
        # Elements in `out_shape` can be i) int ii) -1 iii) scalar Tensor
        # Compute the product of known dimensions in the reshape shape
        known_dims_product = 1
        for dim in out_shape:
            if isinstance(dim, int) and dim == -1:
                continue
            known_dims_product *= dim

        # Compute the total number of elements in the original shape
        total_elements = 1
        input_shape = input_tensor.shape
        for i in range(input_tensor.rank):
            total_elements *= input_shape[i]

        # Infer the dimension
        inferred_dim = total_elements / known_dims_product

        return inferred_dim

    unknown_dim_index = -1
    for i, dim in enumerate(shape):
        if isinstance(dim, int) and dim == -1:
            if unknown_dim_index != -1:
                raise_error(f"Reshape operation size operand can have only one dimension as -1, got shape={shape}.")
            unknown_dim_index = i
    if unknown_dim_index != -1:
        shape = list(shape)
        shape[unknown_dim_index] = compute_unknown_dim(input, shape)

    # we can support infer_len for tp.Shape if the result is rank 1 and the shape is constant
    output_len = None
    if isinstance(input, Shape) and len(shape) == 1 and isinstance(shape[0], int):
        output_len = shape[0]

    return reshape_impl(input, shape, len(shape), output_len)


@dataclass(repr=False)
class Squeeze(BaseTraceOp):

    dims: Tuple[int]
    out_shape: List[int]

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    # Even if given a shape input, the output should not be a shape because the result will not be rank 1.
    # We should permit this, though, since it may be useful to extract a dimension from a shape as a scalar.
    infer_tensor_variants = op_utils.InferVariantPolicies.never_return_shape

    def infer_rank(self):

        if len(self.dims) > 0:
            self.outputs[0].rank = self.inputs[0].rank - len(self.dims)
        else:
            from tripy.backend.mlir.utils import ShapeContext

            input_0_shape = op_utils.get_trace_shape(self.inputs[0])

            def squeeze_shape(shape, indices_to_squeeze):
                # Convert shape to list if it's not already
                shape = list(shape)
                if not indices_to_squeeze:  # If the list is empty, squeeze all dimensions that are 1
                    shape = [dim for dim in shape if dim != 1]
                else:
                    # Sort indices to squeeze in descending order to avoid index shifting issues
                    indices_to_squeeze.sort(reverse=True)
                    for idx in indices_to_squeeze:
                        if shape[idx] == 1:
                            shape.pop(idx)
                        else:
                            raise ValueError(f"Cannot squeeze dimension at index {idx} with value {shape[idx]}")

                return shape

            out_shape = squeeze_shape(input_0_shape, list(self.dims))
            self.outputs[0].rank = len(out_shape)
            self.out_shape = out_shape

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicReshapeOp

        if len(self.dims) > 0:
            select_indices = [i for i in range(inputs[0].rank) if i not in self.dims]
            input_shape = op_utils.get_shape_of_tensor(inputs[0])
            shape_slice = []
            for index in select_indices:
                shape_slice.append(op_utils.slice_rank1_tensor(input_shape, index, reason_details=""))

            output_shape = (
                op_utils.concatenate_tensors(shape_slice, dim=0)
                if len(shape_slice) > 0
                else op_utils.add_constant_tensor_from_list([], inputs[0].device)
            )

        else:
            output_shape = op_utils.add_constant_tensor_from_list(self.out_shape, inputs[0].device)
        DynamicReshapeOp.build([inputs[0], output_shape], outputs)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def squeeze(input: "tripy.Tensor", dims: Union[Tuple, int] = None) -> "tripy.Tensor":
    """
    Returns a new tensor with all specified singleton dimensions of the input tensor removed.

    Args:
        input: The input tensor.
        dims: The singleton dimensions to be removed.
              If this is not provided, all dimensions of size 1 are removed.

    Raises:
        TripyException: If any of the specified dimensions have a size that is not equal to 1.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Squeeze All Dimensions

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, dims=(0, 2))
        assert np.array_equal(cp.from_dlpack(output).get(), np.squeeze(cp.from_dlpack(input).get()))


    .. code-block:: python
        :linenos:
        :caption: Squeeze First Dimension

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, 0)
        assert np.array_equal(cp.from_dlpack(output).get(), np.squeeze(cp.from_dlpack(input).get(), 0))

    .. code-block:: python
        :linenos:
        :caption: Squeeze First And Third Dimension

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, (0, 2))

        assert np.array_equal(cp.from_dlpack(output).get(), np.squeeze(cp.from_dlpack(input).get(), (0, 2)))
    """

    if isinstance(dims, int):
        dims = utils.make_tuple(dims)

    return Squeeze.build([input], dims, None)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def flatten(input: "tripy.Tensor", start_dim: int = 0, end_dim: int = -1) -> "tripy.Tensor":
    """
    Flattens the input tensor from start_dim to end_dim.

    Args:
        input: The input tensor to be flattened.
        start_dim: The first dimension to flatten (default is 0).
        end_dim: The last dimension to flatten (default is -1, which includes the last dimension).

    Returns:
        A flattened tensor.

    .. code-block:: python
        :linenos:
        :caption: Flatten All Dimensions

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.flatten(input)
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(input).get().flatten())

    .. code-block:: python
        :linenos:
        :caption: Flatten Starting from First Dimension

        input = tp.iota((2, 3, 4), dtype=tp.float32)
        output = tp.flatten(input, start_dim=1)
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(input).get().reshape(2, -1))

    .. code-block:: python
        :linenos:
        :caption: Flatten a Specific Range of Dimensions

        input = tp.iota((2, 3, 4, 5), dtype=tp.float32)
        output = tp.flatten(input, start_dim=1, end_dim=2)
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(input).get().reshape(2, -1, 5))
    """

    # Infer the actual dimensions to flatten based on start_dim and end_dim.
    if end_dim < 0:
        end_dim += input.rank

    # Ensure start_dim and end_dim are within the valid range.
    if not (0 <= start_dim < input.rank) or not (start_dim <= end_dim < input.rank):
        raise_error(f"Invalid dimensions: start_dim={start_dim}, end_dim={end_dim}, rank={input.rank}.")

    # Compute the new shape after flattening.
    flattened_dim_size = 1
    for i in range(start_dim, end_dim + 1):
        flattened_dim_size *= input.shape[i]

    from tripy.frontend.shape import Shape

    # The new shape combines the dimensions before start_dim, the flattened dimension, and dimensions after end_dim.
    flattened_shape = input.shape[:start_dim] + Shape(reshape(flattened_dim_size, (1,))) + input.shape[end_dim + 1 :]

    return reshape_impl(input, flattened_shape, len(flattened_shape))
