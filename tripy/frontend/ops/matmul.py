import enum
from dataclasses import dataclass

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.utils import make_tuple

from tripy.common.exception import TripyException


@dataclass
class MatrixMultiplication(BaseOperator):
    """
    Represents a matrix multiplication operation.
    """

    def to_trace_str(self):
        return f"{self.outputs[0].name} = {' @ '.join([inp.name for inp in self.inputs])}"

    def infer_shapes(self):
        # Fix when broadcasting support is added (#25).
        # Note that infer_shapes won't reason about equality of dimensions since shape analysis is done within
        # the mlir tensorrt compiler.

        assert len(self.inputs) == 2, f"{self.__class__.__name__} expects shape for both operands to be present."
        a_shape = self.inputs[0].shape
        b_shape = self.inputs[1].shape

        for i, shape in enumerate([a_shape, b_shape]):
            if len(shape) < 1:
                raise TripyException(
                    f"Operand {i} for {self.__class__.__name__} operation must have number of dims >= 1, got {len(shape)}"
                )

        if len(a_shape) == 1 and len(b_shape) == 1:
            # case 1: both operands are 1-D
            self.contracting_dim = {"lhs": [0], "rhs": [0]}
            self.outputs[0].shape = tuple()
        elif len(a_shape) == 2 and len(b_shape) == 2:
            # case 2: both operands are 2-D
            self.contracting_dim = {"lhs": [1], "rhs": [0]}
            self.outputs[0].shape = (a_shape[0], b_shape[1])
        elif len(a_shape) != len(b_shape):
            raise TripyException("Batched matmul or broadcasting is not implemented, will be fixed by #65.")

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
