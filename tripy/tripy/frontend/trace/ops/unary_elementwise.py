#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tripy import export, dtype_info
from tripy import export, dtype_info
from tripy.frontend.trace.ops.base import BaseTraceOp


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

    # Note: shape inputs will fail because the StableHLO implementations of these ops
    # require float inputs but shapes are always int

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ExpOp, LogOp, RsqrtOp, TanhOp, SineOp, CosineOp, SqrtOp, AbsOp

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


@export.public_api(document_under="tensor_operations")
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def exp(input: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Computes the elementwise exponential of the elements of the input tensor:

    :math:`\text{exp}(x_{i}) = e^{x_{i}}`

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = tp.exp(input)

        assert np.allclose(cp.from_dlpack(output).get(), np.exp(cp.from_dlpack(input).get()))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.EXP)


@export.public_api(document_under="tensor_operations")
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def tanh(input: "tripy.Tensor") -> "tripy.Tensor":
    """
    Computes the elementwise hyperbolic tangent of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = tp.tanh(input)

        assert np.allclose(cp.from_dlpack(output).get(), np.tanh(cp.from_dlpack(input).get()))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.TANH)


@export.public_api(document_under="tensor_operations")
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def sin(input: "tripy.Tensor") -> "tripy.Tensor":
    """
    Computes the elementwise sine of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = tp.sin(input)

        assert np.allclose(cp.from_dlpack(output).get(), np.sin(cp.from_dlpack(input).get()))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.SINE)


@export.public_api(document_under="tensor_operations")
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def cos(input: "tripy.Tensor") -> "tripy.Tensor":
    """
    Computes the elementwise cosine of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32)
        output = tp.cos(input)

        assert np.allclose(cp.from_dlpack(output).get(), np.cos(cp.from_dlpack(input).get()))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.COSINE)


@export.public_api(document_under="tensor_operations")
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def rsqrt(input: "tripy.Tensor") -> "tripy.Tensor":
    """
    Computes the elementwise reciprocal square root of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32) + 1.0
        output = tp.rsqrt(input)

        assert np.allclose(cp.from_dlpack(output).get(), (1.0 / np.sqrt(cp.from_dlpack(input).get())))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.RSQRT)


@export.public_api(document_under="tensor_operations")
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def sqrt(input: "tripy.Tensor") -> "tripy.Tensor":
    """
    Computes the elementwise square root of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(3, dtype=tp.float32) + 1.0
        output = tp.sqrt(input)

        assert np.allclose(cp.from_dlpack(output).get(), (np.sqrt(cp.from_dlpack(input).get())))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.SQRT)


@export.public_api(document_under="tensor_operations")
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def log(input: "tripy.Tensor") -> "tripy.Tensor":
    """
    Computes the elementwise natural logarithm (base e) of the elements of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(1, 3, dtype=tp.float32)
        output = tp.log(input)

        assert np.allclose(cp.from_dlpack(output).get(), (np.log(cp.from_dlpack(input).get())))
    """
    return UnaryElementwise.build([input], UnaryElementwise.Kind.LOG)


@export.public_api(document_under="tensor_operations")
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int8", "int32"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def abs(input: "tripy.Tensor") -> "tripy.Tensor":
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
    from tripy.frontend import Tensor

    return UnaryElementwise.build([input], UnaryElementwise.Kind.ABS)
