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

import enum
from dataclasses import dataclass

from nvtripy import export, wrappers
from nvtripy.frontend.trace.ops.base import BaseTraceOp
import nvtripy.frontend.trace.ops.utils as op_utils


@dataclass(repr=False)
class UnaryElementwise(BaseTraceOp):
    class Kind(enum.Enum):
        EXP = 0
        TANH = 1
        RSQRT = 2
        LOG = 3
        SINE = 4
        COSINE = 5
        SQRT = 6
        ABS = 7

    kind: Kind

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    # Note: shape inputs will fail because the StableHLO implementations of these ops
    # require float inputs but shapes are always int

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import ExpOp, LogOp, RsqrtOp, TanhOp, SineOp, CosineOp, SqrtOp, AbsOp

        OpType = {
            UnaryElementwise.Kind.EXP: ExpOp,
            UnaryElementwise.Kind.TANH: TanhOp,
            UnaryElementwise.Kind.RSQRT: RsqrtOp,
            UnaryElementwise.Kind.LOG: LogOp,
            UnaryElementwise.Kind.SINE: SineOp,
            UnaryElementwise.Kind.COSINE: CosineOp,
            UnaryElementwise.Kind.SQRT: SqrtOp,
            UnaryElementwise.Kind.ABS: AbsOp,
        }[self.kind]
        OpType.build(inputs, outputs)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def exp(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    r"""
    Computes the elementwise exponential of the elements of the input tensor:

    :math:`\text{exp}(x_{i}) = e^{x_{i}}`

    Args:
        input: The input tensor.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = tp.exp(input)

        assert tp.allclose(output, tp.Tensor(np.exp(cp.from_dlpack(input).get())))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.EXP)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def tanh(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Computes the elementwise hyperbolic tangent of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = tp.tanh(input)

        assert tp.allclose(output, tp.Tensor(np.tanh(cp.from_dlpack(input).get())))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.TANH)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def sin(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Computes the elementwise sine of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = tp.sin(input)

        assert tp.allclose(output, tp.Tensor(np.sin(cp.from_dlpack(input).get())))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.SINE)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def cos(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Computes the elementwise cosine of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = tp.cos(input)

        assert tp.allclose(output, tp.Tensor(np.cos(cp.from_dlpack(input).get())))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.COSINE)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8"]},
)
def rsqrt(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Computes the elementwise reciprocal square root of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32) + 1.0
        output = tp.rsqrt(input)

        assert tp.allclose(output, tp.Tensor(1.0 / np.sqrt(cp.from_dlpack(input).get())))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.RSQRT)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8"]},
)
def sqrt(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Computes the elementwise square root of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32) + 1.0
        output = tp.sqrt(input)

        assert tp.allclose(output, tp.Tensor(np.sqrt(cp.from_dlpack(input).get())))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.SQRT)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def log(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Computes the elementwise natural logarithm (base e) of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(1, 3, dtype=tp.float32)
        output = tp.log(input)

        assert tp.allclose(output, tp.Tensor(np.log(cp.from_dlpack(input).get())))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.LOG)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
)
def abs(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    r"""
    Computes the elementwise absolute value of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape with all non-negative entries

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([-1, -2], dtype=tp.int32)
        output = tp.abs(input)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1, 2], dtype=np.float32))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.ABS)
