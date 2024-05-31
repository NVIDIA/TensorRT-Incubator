from dataclasses import dataclass
from typing import Any, Union

import tripy.frontend.trace.ops.utils as op_utils
import tripy.frontend.utils as frontend_utils
from tripy import export
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

    def infer_shapes(self):
        input_shapes = [inp.shape for inp in self.inputs]
        input_shapes = op_utils.get_broadcast_compatible_shapes(self.inputs[0].shape, self.inputs[1].shape)
        bcast_check = op_utils.is_broadcast_compatible(*input_shapes)
        if not bcast_check:
            op_utils.raise_error_io_info(
                self,
                "Input tensors are not broadcast compatible.",
                details=[
                    "Input tensors for operation: '",
                    self.kind.strip(),
                    "' must be broadcast compatible but ",
                ]
                + bcast_check.error_details,
            )
        self.outputs[0].shape = tuple(op_utils.get_broadcast_dim(*d) for d in zip(*input_shapes))

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, self.kind.strip())
        self.outputs[0].dtype = self.inputs[0].dtype

    def broadcast_inputs(self, inputs, outputs):
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.common.datatype import int32
        from tripy.flat_ir.ops import MaxOp

        rank = max(inputs[0].rank, inputs[1].rank)
        with FlatIRTensor.context([f"expand the inputs of '{self.kind.strip()}' to have the same rank"]):
            inputs[0] = op_utils.expand_rank_of_tensor(inputs[0], rank - len(inputs[0].shape))
            inputs[1] = op_utils.expand_rank_of_tensor(inputs[1], rank - len(inputs[1].shape))

        with FlatIRTensor.context([f"broadcast the inputs of '{self.kind.strip()}' to compatible shapes"]):
            shape_of_input0 = op_utils.get_shape_of_tensor(inputs[0])
            shape_of_input1 = op_utils.get_shape_of_tensor(inputs[1])

            # Compute element-wise max of input shapes to get the desired output shape.
            max_output_shape_tensor = FlatIRTensor.build(
                shape=inputs[0].shape,
                rank=rank,
                dtype=int32,
                device=inputs[0].device,
                reason_details=[
                    f"compute the output shape using element-wise max of input shapes {shape_of_input0}, {shape_of_input1} to account for broadcasting."
                ],
            )
            MaxOp.build(
                [shape_of_input0, shape_of_input1],
                [max_output_shape_tensor],
            )

            inputs[0] = op_utils.insert_broadcast(
                inputs[0],
                out_shape=outputs[0].shape,
                out_rank=rank,
                use_dynamic_variant=True,
                shape_of_target_tensor=max_output_shape_tensor,
                tensor_details=f"left operand",
            )

            inputs[1] = op_utils.insert_broadcast(
                inputs[1],
                out_shape=outputs[0].shape,
                out_rank=rank,
                use_dynamic_variant=True,
                shape_of_target_tensor=max_output_shape_tensor,
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
def __add__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise sum.

    Args:
        other: The tensor to add to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

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
def __sub__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise subtraction.

    Args:
        other: The tensor to subtract from this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

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
def __rsub__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise subtraction.

    Args:
        other: The tensor to be subtracted from this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

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
def __pow__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise exponentiation.

    Args:
        other: The tensor by which to exponentiate this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

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
def __rpow__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise exponentiation.

    Args:
        other: The tensor to be exponentiated by this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

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
def __mul__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise multiplication.

    Args:
        other: The tensor by which to multiply this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

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
def __truediv__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise division.

    Args:
        other: The tensor by which to divide this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

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
def __rtruediv__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise division.

    Args:
        other: The tensor to be divided by this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

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
def maximum(lhs: Union["tripy.Tensor", Any], rhs: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise maximum.

    Args:
        lhs: The first input tensor.
        rhs: The second input tensor.
            It must have the same data type as the first input
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

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
def minimum(lhs: Union["tripy.Tensor", Any], rhs: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an elementwise minimum.

    Args:
        lhs: The first input tensor.
        rhs: The second input tensor.
            It must have the same data type as the first input
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and of the same data type as the inputs.

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
def __lt__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'less than' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

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
def __le__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'less than or equal' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

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
def __eq__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs an 'equal' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

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
def __ne__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'not equal' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

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
def __ge__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'greater than or equal' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

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
def __gt__(self, other: Union["tripy.Tensor", Any]) -> "tripy.Tensor":
    """
    Performs a 'greater than' comparison.

    Args:
        other: The tensor to be compared to this one.
            It must have the same data type as this tensor
            and should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape and datatype :class:`tripy.bool`.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([2, 3])
        b = tp.Tensor([3, 1])
        output = b > a

        assert output.eval().data() == [True, False]
    """
    return Comparison.build([self, other], Comparison.Kind.GREATER)
