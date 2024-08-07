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

from dataclasses import dataclass
from typing import Any, Union
import numbers
import tripy.frontend.trace.ops.utils as op_utils
import tripy.frontend.utils as frontend_utils
from tripy import export, dtype_info
from tripy.common import datatype
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class BinaryElementwise(BaseTraceOp):
    class Kind:
        SUM = " + "
        SUB = " - "
        POW = " ** "
        MUL = " * "
        DIV = " / "
        MAXIMUM = "maximum"
        MINIMUM = "minimum"

    kind: str

    def __str__(self):
        if self.kind.startswith(" "):
            op_str = self.kind.join([inp.name for inp in self.inputs])
        else:
            op_str = f"{self.kind}({self.inputs[0].name}, {self.inputs[1].name})"
        return f"{self.outputs[0].name} = {op_str}"

    def infer_shape_output_idxs(self, inputs):
        # permit one input to be a shape but require the output to be a shape
        from tripy.frontend.shape import Shape
        from tripy.utils import Result

        if any(map(lambda t: isinstance(t, Shape), inputs)):
            # if there is a non-shape input, it must be rank 1 or 0 to avoid broadcasting
            if not all(map(lambda t: t.rank <= 1, inputs)):
                invalid_indices_message = ", ".join(
                    map(
                        lambda i: f"Index {i} (rank {inputs[i].rank})",
                        filter(lambda i: inputs[i].rank > 1, range(len(inputs))),
                    )
                )
                return Result.err(
                    [
                        "For binary elementwise operators on Shapes, all inputs must be of rank at most 1.",
                        f"The following inputs have invalid ranks: {invalid_indices_message}",
                    ]
                )
            return Result.ok([0])
        else:
            return Result.ok([])

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, self.kind.strip())
        self.outputs[0].dtype = self.inputs[0].dtype

    def broadcast_inputs(self, inputs, outputs):
        from tripy.common.datatype import int32
        from tripy.flat_ir.ops import MaxOp
        from tripy.flat_ir.tensor import FlatIRTensor

        rank = max(inputs[0].rank, inputs[1].rank)
        with FlatIRTensor.context([f"expand the inputs of '{self.kind.strip()}' to have the same rank"]):
            inputs[0] = op_utils.expand_rank_of_tensor(inputs[0], rank - inputs[0].rank)
            inputs[1] = op_utils.expand_rank_of_tensor(inputs[1], rank - inputs[1].rank)

        with FlatIRTensor.context([f"broadcast the inputs of '{self.kind.strip()}' to compatible shapes"]):
            shape_of_input0 = op_utils.get_shape_of_tensor(inputs[0])
            shape_of_input1 = op_utils.get_shape_of_tensor(inputs[1])

            # Compute element-wise max of input shapes to get the desired output shape.
            output_shape_tensor = op_utils.compute_shape_of_broadcast(
                shape_of_input0,
                shape_of_input1,
                rank,
                shape1_name=f"the shape of the first input {shape_of_input0}",
                shape2_name=f"the shape of the second input {shape_of_input1}",
            )

            with FlatIRTensor.context([f"broadcasting the inputs of '{self.kind.strip()}'"]):
                inputs[0] = op_utils.insert_broadcast(
                    inputs[0],
                    out_rank=rank,
                    use_dynamic_variant=True,
                    shape_of_target_tensor=output_shape_tensor,
                    tensor_details=f"left operand",
                )

                inputs[1] = op_utils.insert_broadcast(
                    inputs[1],
                    out_rank=rank,
                    use_dynamic_variant=True,
                    shape_of_target_tensor=output_shape_tensor,
                    tensor_details=f"right operand",
                )

        return inputs

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import AddOp, DivideOp, MaxOp, MinOp, MulOp, PowOp, SubtractOp

        inputs = self.broadcast_inputs(inputs, outputs)
        OpType = {
            BinaryElementwise.Kind.SUM: AddOp,
            BinaryElementwise.Kind.POW: PowOp,
            BinaryElementwise.Kind.MUL: MulOp,
            BinaryElementwise.Kind.SUB: SubtractOp,
            BinaryElementwise.Kind.DIV: DivideOp,
            BinaryElementwise.Kind.MAXIMUM: MaxOp,
            BinaryElementwise.Kind.MINIMUM: MinOp,
        }[self.kind]
        OpType.build(inputs, outputs)


@dataclass(repr=False)
class Comparison(BinaryElementwise):
    class Kind:
        class KindElem(str):
            def __new__(cls, content, compare_direction):
                instance = super().__new__(cls, content)
                instance.compare_direction = compare_direction
                return instance

        LESS = KindElem(" < ", "LT")
        LESS_EQUAL = KindElem(" <= ", "LE")
        EQUAL = KindElem(" == ", "EQ")
        NOT_EQUAL = KindElem(" != ", "NE")
        GREATER_EQUAL = KindElem(" >= ", "GE")
        GREATER = KindElem(" > ", "GT")

    kind: Kind.KindElem

    # the result of a comparison will be bool, so do not wrap
    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, self.kind.strip())
        self.outputs[0].dtype = datatype.bool

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import CompareOp

        inputs = self.broadcast_inputs(inputs, outputs)
        CompareOp.build(inputs, outputs, compare_direction=self.kind.compare_direction)


@TENSOR_METHOD_REGISTRY("__add__")
@TENSOR_METHOD_REGISTRY("__radd__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def __add__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise sum.

    Args:
        self: Tensor to be added to other.
        other: The tensor to add to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([1, 2])
        b = tp.Tensor([2, 3])
        output = a + b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([3, 5]))
    """
    return BinaryElementwise.build([self, other], BinaryElementwise.Kind.SUM)


@TENSOR_METHOD_REGISTRY("__sub__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def __sub__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise subtraction.

    Args:
        self: Tensor to be subtracted by other.
        other: The tensor to subtract from this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 2])
        output = a - b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1, 1]))
    """
    return BinaryElementwise.build([self, other], BinaryElementwise.Kind.SUB)


@TENSOR_METHOD_REGISTRY("__rsub__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
def __rsub__(self: numbers.Number, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise subtraction.

    Args:
        self: Tensor to be subtracted by other.
        other: The tensor to be subtracted from this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = 1
        b = tp.Tensor([1, 2])
        output = a - b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([0, -1]))
    """
    return BinaryElementwise.build([other, self], BinaryElementwise.Kind.SUB)


@TENSOR_METHOD_REGISTRY("__pow__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def __pow__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise exponentiation.

    Args:
        self: Tensor to be exponentiated by other.
        other: The tensor by which to exponentiate this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([1.0, 2.0])
        b = tp.Tensor([2.0, 3.0])
        output = a ** b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1, 8]))
    """
    return BinaryElementwise.build([self, other], BinaryElementwise.Kind.POW)


@TENSOR_METHOD_REGISTRY("__rpow__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
def __rpow__(self: numbers.Number, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise exponentiation.

    Args:
        self: Tensor to be exponentiated by other.
        other: The tensor to be exponentiated by this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = 2.0
        b = tp.Tensor([2.0, 3.0])
        output = a ** b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([4.0, 8.0]))
    """
    return BinaryElementwise.build([other, self], BinaryElementwise.Kind.POW)


@TENSOR_METHOD_REGISTRY("__mul__")
@TENSOR_METHOD_REGISTRY("__rmul__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def __mul__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise multiplication.

    Args:
        self: Tensor to be multiplied by other.
        other: The tensor by which to multiply this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([1.0, 2.0])
        b = tp.Tensor([2.0, 3.0])
        output = a * b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 6.0]))
    """
    return BinaryElementwise.build([self, other], BinaryElementwise.Kind.MUL)


@TENSOR_METHOD_REGISTRY("__truediv__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def __truediv__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise division.

    Args:
        self: Tensor to be divided by other.
        other: The tensor by which to divide this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([4.0, 6.0])
        b = tp.Tensor([2.0, 3.0])
        output = a / b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 2.0]))
    """
    return BinaryElementwise.build([self, other], BinaryElementwise.Kind.DIV)


@TENSOR_METHOD_REGISTRY("__rtruediv__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
def __rtruediv__(self: numbers.Number, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise division.

    Args:
        self: Tensor to be subtracted by other.
        other: The tensor to be divided by this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = 6.0
        b = tp.Tensor([2.0, 3.0])
        output = a / b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([3.0, 2.0]))
    """
    return BinaryElementwise.build([other, self], BinaryElementwise.Kind.DIV)


@export.public_api(document_under="tensor_operations")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("lhs", "rhs")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"lhs": "T1", "rhs": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def maximum(lhs: Union["tripy.Tensor", Any], rhs: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise maximum.

    Args:
        lhs: The first input tensor.
        rhs: The second input tensor.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([1.0, 6.0])
        b = tp.Tensor([2.0, 3.0])
        output = tp.maximum(a, b)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 6.0]))
    """
    return BinaryElementwise.build([lhs, rhs], BinaryElementwise.Kind.MAXIMUM)


@export.public_api(document_under="tensor_operations")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("lhs", "rhs")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"lhs": "T1", "rhs": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def minimum(lhs: Union["tripy.Tensor", Any], rhs: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise minimum.

    Args:
        lhs: The first input tensor.
        rhs: The second input tensor.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([1.0, 6.0])
        b = tp.Tensor([2.0, 3.0])
        output = tp.minimum(a, b)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1.0, 3.0]))
    """
    return BinaryElementwise.build([lhs, rhs], BinaryElementwise.Kind.MINIMUM)


@TENSOR_METHOD_REGISTRY("__lt__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"], "T2": ["bool"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T2"},
)
def __lt__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'less than' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 5])
        output = b < a
        # TODO(#26): replace with cp.from_dlpack(output).get() after MLIR-TRT can handle i1 dtype's allocation

        assert output.eval().data() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.LESS)


@TENSOR_METHOD_REGISTRY("__le__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"], "T2": ["bool"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T2"},
)
def __le__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'less than or equal' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 5])
        output = b <= a

        assert output.eval().data() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.LESS_EQUAL)


@TENSOR_METHOD_REGISTRY("__eq__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"], "T2": ["bool"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T2"},
)
def __eq__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an 'equal' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 5])
        output = b == a

        assert output.eval().data() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.EQUAL)


@TENSOR_METHOD_REGISTRY("__ne__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"], "T2": ["bool"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T2"},
)
def __ne__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'not equal' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([1, 3])
        output = b != a

        assert output.eval().data() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.NOT_EQUAL)


@TENSOR_METHOD_REGISTRY("__ge__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"], "T2": ["bool"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T2"},
)
def __ge__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'greater than or equal' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([2, 1])
        output = b >= a

        assert output.eval().data() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.GREATER_EQUAL)


@TENSOR_METHOD_REGISTRY("__gt__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@dtype_info.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"], "T2": ["bool"]},
    dtype_constraints={"self": "T1", "other": "T1", dtype_info.RETURN_VALUE: "T2"},
)
def __gt__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'greater than' comparison.

    Args:
        self: Tensor to be compared with other.
        other: The tensor to be compared to this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([3, 1])
        output = b > a

        assert output.eval().data() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.GREATER)
