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
from typing import Any, Union

import tripy.frontend.trace.ops.utils as op_utils
import tripy.frontend.utils as frontend_utils
from tripy import constraints, export
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
        FLOOR_DIV = " // "
        MOD = " % "
        MAXIMUM = "maximum"
        MINIMUM = "minimum"

    kind: str

    def __str__(self):
        if self.kind.startswith(" "):
            op_str = self.kind.join([inp.name for inp in self.inputs])
        else:
            op_str = f"{self.kind}({self.inputs[0].name}, {self.inputs[1].name})"
        return f"{self.outputs[0].name} = {op_str}"

    def infer_tensor_variants(self, inputs):
        # permit one input to be a shape but require the output to be a shape
        from tripy.frontend.shape import Shape, ShapeScalar
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
            return Result.ok([Shape])
        elif all(map(lambda t: isinstance(t, ShapeScalar), inputs)):
            # Binary operation on ShapeScalar should yield another ShapeScalar.
            return Result.ok([ShapeScalar])
        else:
            return Result.ok([None])

    def infer_len(self):
        # For the shape case, the result will be broadcast to the max of the input shapes
        input_lengths = []
        for inp in self.inputs:
            shape = op_utils.get_trace_shape(inp)
            if len(shape) != 0:
                input_lengths.append(shape[0])
        return [max(input_lengths)]

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def broadcast_inputs(self, inputs):
        from tripy.flat_ir.tensor import FlatIRTensor

        rank = max(inputs[0].rank, inputs[1].rank)
        with FlatIRTensor.context([f"expand the inputs of '{self.kind.strip()}' to have the same rank"]):
            broadcasted_input_0 = op_utils.expand_rank_of_tensor(inputs[0], rank - inputs[0].rank)
            broadcasted_input_1 = op_utils.expand_rank_of_tensor(inputs[1], rank - inputs[1].rank)

        with FlatIRTensor.context([f"broadcast the inputs of '{self.kind.strip()}' to compatible shapes"]):
            shape_of_input0 = op_utils.get_shape_of_tensor(broadcasted_input_0)
            shape_of_input1 = op_utils.get_shape_of_tensor(broadcasted_input_1)

            # Compute element-wise max of input shapes to get the desired output shape.
            output_shape_tensor = op_utils.compute_shape_of_broadcast(
                shape_of_input0,
                shape_of_input1,
                rank,
                shape1_name=f"the shape of the first input {shape_of_input0}",
                shape2_name=f"the shape of the second input {shape_of_input1}",
            )

            with FlatIRTensor.context([f"broadcasting the inputs of '{self.kind.strip()}'"]):
                broadcasted_input_0 = op_utils.insert_broadcast(
                    broadcasted_input_0,
                    out_rank=rank,
                    shape_of_target_tensor=output_shape_tensor,
                    tensor_details=f"left operand",
                )

                broadcasted_input_1 = op_utils.insert_broadcast(
                    broadcasted_input_1,
                    out_rank=rank,
                    shape_of_target_tensor=output_shape_tensor,
                    tensor_details=f"right operand",
                )

        return [broadcasted_input_0, broadcasted_input_1]

    @frontend_utils.make_function
    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import AddOp, DivideOp, FloorOp, MaxOp, MinOp, MulOp, PowOp, SubtractOp
        from tripy.flat_ir.tensor import FlatIRTensor

        broadcasted_inputs = self.broadcast_inputs(inputs)

        if self.kind == BinaryElementwise.Kind.FLOOR_DIV:
            # First apply DivideOp
            divide_out = FlatIRTensor.build(
                shape=outputs[0].shape,
                rank=outputs[0].rank,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=["Intermediate output of division operator for FLOOR_DIV operation."],
            )
            DivideOp.build(broadcasted_inputs, [divide_out])
            # Then apply FloorOp to the result of the division
            FloorOp.build([divide_out], outputs)
        elif self.kind == BinaryElementwise.Kind.MOD:
            # Step 1: Perform DivideOp
            divide_out = FlatIRTensor.build(
                shape=outputs[0].shape,
                rank=outputs[0].rank,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=["Intermediate output of division operator for MOD operation."],
            )
            DivideOp.build(broadcasted_inputs, [divide_out])

            # Step 2: Apply FloorOp
            floor_out = FlatIRTensor.build(
                shape=outputs[0].shape,
                rank=outputs[0].rank,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=["Intermediate output of Floor operation for MOD operation."],
            )
            FloorOp.build([divide_out], [floor_out])

            # Step 3: Multiply divisor with floored division result (FloorOp output)
            multiply_out = FlatIRTensor.build(
                shape=outputs[0].shape,
                rank=outputs[0].rank,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=["Intermediate output of Multiply operation for MOD operation."],
            )
            MulOp.build([broadcasted_inputs[1], floor_out], [multiply_out])

            # Step 4: Subtract result from dividend (broadcasted_inputs[0]) to get modulus
            SubtractOp.build([broadcasted_inputs[0], multiply_out], outputs)
        else:
            OpType = {
                BinaryElementwise.Kind.SUM: AddOp,
                BinaryElementwise.Kind.POW: PowOp,
                BinaryElementwise.Kind.MUL: MulOp,
                BinaryElementwise.Kind.SUB: SubtractOp,
                BinaryElementwise.Kind.DIV: DivideOp,
                BinaryElementwise.Kind.MAXIMUM: MaxOp,
                BinaryElementwise.Kind.MINIMUM: MinOp,
                BinaryElementwise.Kind.FLOOR_DIV: DivideOp,
            }[self.kind]
            OpType.build(broadcasted_inputs, outputs)


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
    infer_tensor_variants = op_utils.InferVariantPolicies.never_return_shape

    def infer_dtypes(self):
        self.outputs[0].dtype = datatype.bool

    @frontend_utils.make_function
    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import CompareOp

        inputs = self.broadcast_inputs(inputs)
        CompareOp.build(inputs, outputs, compare_direction=self.kind.compare_direction)


@TENSOR_METHOD_REGISTRY("__add__")
@TENSOR_METHOD_REGISTRY("__radd__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T1"},
    aliases=["__radd__"],
)
def __add__(
    self: "tripy.types.TensorLike",
    other: "tripy.types.TensorLike",
) -> "tripy.Tensor":
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
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T1"},
)
def __sub__(
    self: "tripy.types.TensorLike",
    other: "tripy.types.TensorLike",
) -> "tripy.Tensor":
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
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
    dtype_constraints={"other": "T1", constraints.RETURN_VALUE: "T1"},
)
def __rsub__(self: "tripy.types.NestedNumberSequence", other: "tripy.types.TensorLike") -> "tripy.Tensor":
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
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8"]},
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T1"},
)
def __pow__(
    self: "tripy.types.TensorLike",
    other: "tripy.types.TensorLike",
) -> "tripy.Tensor":
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
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8"]},
    dtype_constraints={"other": "T1", constraints.RETURN_VALUE: "T1"},
)
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
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T1"},
    aliases=["__rmul__"],
)
def __mul__(
    self: "tripy.types.TensorLike",
    other: "tripy.types.TensorLike",
) -> "tripy.Tensor":
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
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T1"},
)
def __truediv__(
    self: "tripy.types.TensorLike",
    other: "tripy.types.TensorLike",
) -> "tripy.Tensor":
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
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
    dtype_constraints={"other": "T1", constraints.RETURN_VALUE: "T1"},
)
def __rtruediv__(self: numbers.Number, other: "tripy.types.TensorLike") -> "tripy.Tensor":
    """
    Performs an elementwise division.

    Args:
        self: Tensor to be divided by other.
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


@TENSOR_METHOD_REGISTRY("__floordiv__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64"]},
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T1"},
)
def __floordiv__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise floor division.

    Args:
        self: Tensor to be floor-divided by other.
        other: The tensor by which to floor-divide this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([4.0, 6.0])
        b = tp.Tensor([3.0, 4.0])
        output = a // b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1.0, 1.0]))
    """
    from tripy.common.datatype import int32
    from tripy.frontend.trace.ops.cast import cast

    return cast(cast(BinaryElementwise.build([self, other], BinaryElementwise.Kind.DIV), int32), self.dtype)
    # Use the below code when https://github.com/NVIDIA/TensorRT-Incubator/issues/208 is fixed
    # return BinaryElementwise.build([self, other], BinaryElementwise.Kind.FLOOR_DIV)


@TENSOR_METHOD_REGISTRY("__rfloordiv__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64"]},
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T1"},
)
def __rfloordiv__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise floor division.

    Args:
        self: Tensor to be floor-divided by other.
        other: The tensor to be floor-divided by this one.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = 2
        b = tp.Tensor([2.0, 3.0])
        output = a // b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1.0, 0.0]))
    """
    from tripy.common.datatype import int32
    from tripy.frontend.trace.ops.cast import cast

    return cast(cast(BinaryElementwise.build([other, self], BinaryElementwise.Kind.DIV), int32), self.dtype)
    # Use the below code when https://github.com/NVIDIA/TensorRT-Incubator/issues/208 is fixed
    # return BinaryElementwise.build([other, self], BinaryElementwise.Kind.FLOOR_DIV)


@TENSOR_METHOD_REGISTRY("__mod__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8"]},
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T1"},
)
def __mod__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a modulo operation.

    Args:
        self: The tensor to be divided by `other`.
        other: The tensor by which to divide `self`.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape containing the result of the modulo operation.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([4.0, 6.0])
        b = tp.Tensor([3.0, 4.0])
        output = a % b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1.0, 2.0]))
    """
    return BinaryElementwise.build([self, other], BinaryElementwise.Kind.MOD)


@TENSOR_METHOD_REGISTRY("__rmod__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8"]},
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T1"},
)
def __rmod__(self: Union["tripy.Tensor", Any], other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a modulo operation.

    Args:
        self: The tensor to be divided by `other`.
        other: The tensor by which to divide `self`.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape containing the result of the modulo operation.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([4.0, 6.0])
        output = 2 % a

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 2.0]))
    """
    return BinaryElementwise.build([other, self], BinaryElementwise.Kind.MOD)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"lhs": "T1", "rhs": "T1", constraints.RETURN_VALUE: "T1"},
)
def maximum(lhs: "tripy.Tensor", rhs: "tripy.Tensor") -> "tripy.Tensor":
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


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"]},
    dtype_constraints={"lhs": "T1", "rhs": "T1", constraints.RETURN_VALUE: "T1"},
)
def minimum(lhs: "tripy.Tensor", rhs: "tripy.Tensor") -> "tripy.Tensor":
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
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T2"},
)
def __lt__(
    self: "tripy.types.TensorLike",
    other: "tripy.types.TensorLike",
) -> "tripy.Tensor":
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

        assert output.tolist() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.LESS)


@TENSOR_METHOD_REGISTRY("__le__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T2"},
)
def __le__(
    self: "tripy.types.TensorLike",
    other: "tripy.types.TensorLike",
) -> "tripy.Tensor":
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

        assert output.tolist() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.LESS_EQUAL)


@TENSOR_METHOD_REGISTRY("__eq__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T2"},
)
def __eq__(
    self: "tripy.types.TensorLike",
    other: "tripy.types.TensorLike",
) -> "tripy.Tensor":
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

        assert output.tolist() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.EQUAL)


@TENSOR_METHOD_REGISTRY("__ne__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T2"},
)
def __ne__(self: "tripy.types.TensorLike", other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
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

        assert output.tolist() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.NOT_EQUAL)


@TENSOR_METHOD_REGISTRY("__ge__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T2"},
)
def __ge__(
    self: "tripy.types.TensorLike",
    other: "tripy.types.TensorLike",
) -> "tripy.Tensor":
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

        assert output.tolist() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.GREATER_EQUAL)


@TENSOR_METHOD_REGISTRY("__gt__")
@frontend_utils.convert_inputs_to_tensors(sync_arg_types=[("self", "other")])
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    dtype_constraints={"self": "T1", "other": "T1", constraints.RETURN_VALUE: "T2"},
)
def __gt__(
    self: "tripy.types.TensorLike",
    other: "tripy.types.TensorLike",
) -> "tripy.Tensor":
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

        assert output.tolist() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.GREATER)
