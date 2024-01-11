import enum
from dataclasses import dataclass

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.util import make_tuple

from tripy.common.exception import TripyException


@dataclass
class MatrixMultiplication(BaseOperator):
    """
    Represents a matrix multiplication operation.
    """

    def to_trace_str(self, input_names, output_names):
        assert len(output_names) == 1, f"{self.__class__.__name__} should have exactly one output!"
        return f"{output_names[0]} = {' @ '.join(input_names)}"

    def infer_shapes(self, input_shapes):
        # Fix when broadcasting support is added (#25).
        # Note that infer_shapes won't reason about equality of dimensions since shape analysis is done within
        # the mlir tensorrt compiler.

        assert len(input_shapes) == 2, f"{self.__class__.__name__} expects shape for both operands to be present."
        a_shape = input_shapes[0]
        b_shape = input_shapes[1]

        for i, operand in enumerate(input_shapes):
            if len(operand) < 1:
                raise TripyException(
                    f"Operand {i} for {self.__class__.__name__} operation must have number of dims >= 1, got {len(operand)}"
                )

        # case 1: both operands are 1-D
        if len(a_shape) == 1 and len(b_shape) == 1:
            self.contracting_dim = {"lhs": [0], "rhs": [0]}
            return [make_tuple([])]

        # case 2: both operands are 2-D
        if len(a_shape) == 2 and len(b_shape) == 2:
            self.contracting_dim = {"lhs": [1], "rhs": [0]}
            return [make_tuple([a_shape[0], b_shape[1]])]

        if len(a_shape) != len(b_shape):
            raise TripyException("Batched matmul or broadcasting is not implemented, will be fixed by #65.")

    def infer_dtypes(self, input_dtypes):
        assert (
            input_dtypes[0] == input_dtypes[1]
        ), f"Input data types for BinaryElementwise must match. Got: {input_dtypes[0]} and {input_dtypes[1]}"
        return [input_dtypes[0]]

    def to_flat_ir(self, flat_ir):
        from tripy.flat_ir.ops.dot import DotOp

        flat_ir.add_op(self, DotOp, self.inputs, self.outputs, contracting_dim=self.contracting_dim)


@TENSOR_METHOD_REGISTRY("__matmul__")
def matmul(self: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    """
    Performs matrix multiplication between two tensors.

    This operation follows numpy like behavior for arguments.
    If both tensors are 1-D, scalar product is returned.
    If both tensors are 2-D, regular matrix-matrix multiplication is returned.
    If either argument is 1-D, 1 dimension is inserted and matrix
    multiplication is performed with relevant broadcast of dimension.

    Args:
        other: The tensor to multiply to this one.

    Returns:
        Result of matrix multiplication between two tensors.

    Example:
    ::

        import numpy as np

        a_np = np.ones((2,3), dtype=np.float32)
        b_np = np.ones((3,2), dtype=np.float32)
        a = tp.Tensor(a_np)
        b = tp.Tensor(b_np)

        out = a @ b
        assert (out.numpy() == (a_np @ b_np)).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        MatrixMultiplication,
    )
