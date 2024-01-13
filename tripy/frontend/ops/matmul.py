from dataclasses import dataclass

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
import tripy.frontend.ops.utils as op_utils


@dataclass
class MatrixMultiplication(BaseOperator):
    """
    Represents a matrix multiplication operation.
    """

    def __str__(self):
        return f"{self.outputs[0].name} = {' @ '.join([inp.name for inp in self.inputs])}"

    def infer_shapes(self):
        # Fix when broadcasting support is added (#25).
        # Note that infer_shapes won't reason about equality of dimensions since shape analysis is done within
        # the mlir tensorrt compiler.

        assert len(self.inputs) == 2, f"MatrixMultiplication expects exactly two inputs."
        a_shape = self.inputs[0].shape
        b_shape = self.inputs[1].shape

        for index, shape in enumerate([a_shape, b_shape]):
            if len(shape) < 1:
                op_utils.raise_error_io_info(
                    self,
                    "Input tensors must have at least 1 dimension.",
                    details=[
                        f"Inputs for operation: '@' must have at least one dimension, but input {index} has shape: {shape} which has fewer than 1 dimension."
                    ],
                )

        if len(a_shape) == 1 and len(b_shape) == 1:
            # case 1: both operands are 1-D
            op_utils.check_input_shapes_match(self, "@")

            self.contracting_dim = {"lhs": [0], "rhs": [0]}
            self.outputs[0].shape = tuple()

        elif len(a_shape) == 2 and len(b_shape) == 2:
            if a_shape[1] != b_shape[0]:
                op_utils.raise_error_io_info(
                    self,
                    "Incompatible input shapes.",
                    details=[
                        f"For operation: '@', the second dimension of input 0 (shape: {a_shape}) must match the first "
                        f"dimension of input 1 (shape: {b_shape}) but got: {a_shape[1]} and {b_shape[0]}"
                    ],
                )
            # case 2: both operands are 2-D
            self.contracting_dim = {"lhs": [1], "rhs": [0]}
            self.outputs[0].shape = (a_shape[0], b_shape[1])
        elif len(a_shape) != len(b_shape):
            assert False, "Batched matmul or broadcasting is not implemented, will be fixed by #65."

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, "@")
        self.outputs[0].dtype = self.inputs[0].dtype

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

        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.ones((3, 2), dtype=tp.float32)

        out = a @ b
        print(out)
        assert (out.numpy() == (np.ones((2,3), dtype=np.float32) @ np.ones((3,2), dtype=np.float32))).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        MatrixMultiplication,
    )
