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

import math
import enum
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from tripy import export, constraints
from tripy.common import datatype
from tripy.frontend.trace.ops.base import BaseTraceOp
import tripy.frontend.trace.ops.utils as op_utils
from tripy.utils import make_list


@dataclass(repr=False)
class Reduce(BaseTraceOp):
    class Kind(enum.Enum):
        def __init__(self, op, init_value):
            self.op = op
            self.init_value = init_value

        SUM = "sum", 0
        MAX = "max", 0
        MUL = "mul", 1
        AND = "and", True
        OR = "or", False

    dim: Sequence[int]
    kind: Kind

    # if the input is a shape, the output is likely not going to be rank 1 so we should not wrap as a shape
    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_rank(self):
        if self.dim is None:
            self.dim = list(range(self.inputs[0].rank))
            self.outputs[0].rank = 0
        else:
            self.dim = make_list(self.dim)
            self.dim = [idx if idx >= 0 else idx + self.inputs[0].rank for idx in self.dim]
            self.outputs[0].rank = self.inputs[0].rank - len(self.dim)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConstantOp, ReduceOp
        from tripy.flat_ir.tensor import FlatIRTensor

        init_value = self.kind.init_value
        init_const = FlatIRTensor.build(
            shape=(),
            rank=0,
            dtype=outputs[0].dtype,
            device=outputs[0].device,
            reason_details=[
                f"create the constant value tensor (containing {init_value}) for the initial value of a '{self.kind.op}' operation"
            ],
        )
        ConstantOp.build([], [init_const], data=init_value)

        ReduceOp.build([inputs[0], init_const], outputs, reduce_mode=self.kind.op, reduce_dims=self.dim)


@dataclass(repr=False)
class ArgMinMax(Reduce):
    class Kind:
        ARG_MAX = "argmax"
        ARG_MIN = "argmin"

    dim: Sequence[int]
    kind: Kind

    def infer_dtypes(self):
        self.outputs[0].dtype = datatype.int32

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ArgMinMaxOp, ConstantOp
        from tripy.flat_ir.tensor import FlatIRTensor

        init_val_const = FlatIRTensor.build(
            shape=[],
            rank=0,
            dtype=inputs[0].dtype,
            device=outputs[0].device,
            reason_details=[f"create the constant value tensor for the initial value of a '{self.kind}' operation"],
        )
        init_idx_const = FlatIRTensor.build(
            shape=[],
            rank=0,
            dtype=outputs[0].dtype,
            device=outputs[0].device,
            reason_details=[
                f"create the constant value tensor for the initial index value of a '{self.kind}' operation"
            ],
        )

        ConstantOp.build([], [init_val_const], data=0)
        ConstantOp.build([], [init_idx_const], data=0)

        ArgMinMaxOp.build(
            [inputs[0], inputs[1], init_val_const, init_idx_const],
            outputs,
            reduce_mode=self.kind,
            reduce_dims=self.dim,
        )


def _reduce_impl(input: "tripy.Tensor", kind: Reduce.Kind, dim: Union[int, Sequence], keepdim: bool):
    from tripy.frontend.trace.ops.unsqueeze import unsqueeze
    from tripy.frontend.trace.ops.reshape import reshape

    out = Reduce.build([input], dim, kind)
    if keepdim:
        if dim is None:
            out = reshape(out, (1,) * input.rank)
        else:
            # Custom comparison function ensures negatives are sorted in decreasing order, otherwise increasing.
            # e.g, [-2, 0, -1, 2] is sorted as [-1, -2, 0, 2].
            for d in sorted(make_list(dim), key=lambda x: (0, -x) if x < 0 else (1, x)):
                out = unsqueeze(out, d)

    return out


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "int32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def sum(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "tripy.Tensor":
    """
    Returns a new tensor containing the sum of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.sum(input, 0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.sum(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(input, Reduce.Kind.SUM, dim, keepdim)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["bool"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def all(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "tripy.Tensor":
    """
    Returns a new tensor containing the logical AND of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new bool tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([True, True], dtype=tp.bool)
        out = tp.all(input)
        assert bool(out)
    """
    return _reduce_impl(input, Reduce.Kind.AND, dim, keepdim)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["bool"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def any(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "tripy.Tensor":
    """
    Returns a new tensor containing the logical OR of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new bool tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([True, False], dtype=tp.bool)
        out = tp.any(input)
        assert bool(out)
    """
    return _reduce_impl(input, Reduce.Kind.OR, dim, keepdim)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "int32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def max(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "tripy.Tensor":
    """
    Returns a new tensor containing the maximum of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.max(input, 0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.max(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(input, Reduce.Kind.MAX, dim, keepdim)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "int32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def prod(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "tripy.Tensor":
    """
    Returns a new tensor containing the product of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.prod(input, 0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.prod(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(input, Reduce.Kind.MUL, dim, keepdim)


def mean_impl(tensor: "tripy.Tensor", dim: Union[int, Sequence] = None, keepdim: bool = False, apply_to_divisor=None):
    from tripy.frontend.trace.ops.cast import cast

    sum_val = sum(tensor, dim=dim, keepdim=keepdim)

    # compute number of elements in the array and divide by number of elements in dims
    num_elements = math.prod(tensor.shape if dim is None else [tensor.shape[d] for d in make_list(dim)])

    if apply_to_divisor:
        num_elements = apply_to_divisor(num_elements)

    return sum_val / (cast(num_elements, sum_val.dtype))


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "int32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def mean(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "tripy.Tensor":
    """
    Returns a new tensor containing the mean of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        mean of the input tensor

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.mean(input, dim=1, keepdim=True)

        assert np.array_equal(cp.from_dlpack(output).get(), np.mean(np.arange(6, dtype=np.float32).reshape((2, 3)), axis=1, keepdims=True))
    """
    return mean_impl(input, dim, keepdim)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def var(
    input: "tripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False, correction: int = 1
) -> "tripy.Tensor":
    r"""
    Returns a new tensor containing the variance of the elements of the input tensor along the specified dimension.

    The variance along a dimension is defined as:

    :math:`\sigma^2 = \Large \frac{1}{max(0, N - \text{correction})} \large \sum_{i=1}^N (x_i - \bar{x})^2`

    where :math:`N` is the length of the dimension, :math:`x_i` is the :math:`i^{th}` element along the dimension,
    and :math:`\bar{x}` is the mean.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.
        correction: Defaults to Bessel's correction.

    Returns:
        variance of the input tensor

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.var(input, dim=1, keepdim=True)

        torch_input = torch.arange(6, dtype=torch.float32).reshape((2, 3)) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), np.from_dlpack(torch_input.var(dim=1, keepdim=True)))
    """
    from tripy.frontend import Tensor
    from tripy.frontend.trace.ops.binary_elementwise import maximum

    mean_val = mean(input, dim=dim, keepdim=dim is not None)
    sub = (input - mean_val) ** 2.0
    return mean_impl(
        sub, dim=dim, keepdim=keepdim, apply_to_divisor=lambda x: maximum(x - Tensor(correction), Tensor(0))
    )


def _arg_min_max_impl(tensor: "tripy.Tensor", kind: ArgMinMax.Kind, dim: Optional[int], keepdim: bool):
    from tripy.frontend.trace.ops.iota import iota_like
    from tripy.frontend.trace.ops.reshape import reshape
    from tripy.frontend.trace.ops.unsqueeze import unsqueeze

    input_rank = tensor.rank
    if dim is None:
        tensor = reshape(tensor, (-1,))
    indices = iota_like(tensor, dim if dim else 0, datatype.int32)
    out = ArgMinMax.build([tensor, indices], dim, kind)
    if keepdim:
        if dim is None:
            out = reshape(out, (1,) * input_rank)
        else:
            out = unsqueeze(out, dim)
    return out


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int32", "bool", "int8"], "T2": ["int32"]},
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T2"},
)
def argmax(input: "tripy.Tensor", dim: Optional[int] = None, keepdim: bool = False) -> "tripy.Tensor":
    """
    Returns a new tensor containing the indices of maximum values of the input tensor along the specified dimension.
    If there are multiple maximum values, then the indices of the first maximum value are returned.

    Args:
        input: The input tensor.
        dim: The dimension along which to reduce.
            If this is not provided, the argmax indice of the flattened input is returned.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.argmax(input, 0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.argmax(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _arg_min_max_impl(input, ArgMinMax.Kind.ARG_MAX, dim, keepdim)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int32", "bool", "int8"], "T2": ["int32"]},
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T2"},
)
def argmin(input: "tripy.Tensor", dim: Optional[int] = None, keepdim: bool = False) -> "tripy.Tensor":
    """
    Returns a new tensor containing the indices of minimum values of the input tensor along the specified dimension.
    If there are multiple minimum values, then the indices of the first minimum value are returned.

    Args:
        input: The input tensor.
        dim: The dimension along which to reduce.
            If this is not provided, the argmin indice of the flattened input is returned.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.argmin(input, 0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.argmin(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _arg_min_max_impl(input, ArgMinMax.Kind.ARG_MIN, dim, keepdim)
