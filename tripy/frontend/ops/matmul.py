from dataclasses import dataclass
import copy

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

    def get_operand_shape_after_broadcast(self, a_shape, b_shape):
        # Split the a and b shape into batch and matrix dims.
        a_batch, a_matrix = a_shape[:-2], a_shape[-2:]
        b_batch, b_matrix = b_shape[:-2], b_shape[-2:]

        # Broadcasting batch dimensions
        batch_shapes = op_utils.get_broadcast_compatible_shapes(a_batch, b_batch)
        batch_shapes = tuple(op_utils.get_broadcast_dim(*d) for d in zip(*batch_shapes))
        a_shape = batch_shapes + a_matrix
        b_shape = batch_shapes + b_matrix
        return a_shape, b_shape

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
            self.batching_dim = {"lhs": [], "rhs": []}
            self.contracting_dim = {"lhs": [0], "rhs": [0]}
            self.outputs[0].shape = tuple()
        else:
            # stablehlo dot_general requires same number of batching dims for lhs, rhs.

            def get_contracting_dim(shape, lhs=True):
                if lhs or len(shape) == 1:
                    return [len(shape) - 1]
                else:
                    return [len(shape) - 2]

            def get_batch_indices(shape):
                return list(range(len(shape) - 2))

            def get_output_shape(shape_a, shape_b):
                # Determine the indices based on the length of shape_a and shape_b
                shape_a_index = () if len(shape_a) == 1 else (shape_a[-2],)
                shape_b_index = () if len(shape_b) == 1 else (shape_b[-1],)

                final_shape = shape_a[:-2]
                if shape_a_index != ():
                    final_shape += shape_a_index

                if shape_b_index != ():
                    final_shape += shape_b_index

                return final_shape

            if a_shape[get_contracting_dim(a_shape)[0]] != b_shape[get_contracting_dim(b_shape, lhs=False)[0]]:
                op_utils.raise_error_io_info(
                    self,
                    "Incompatible input shapes.",
                    details=[
                        f"For operation: '@', the second dimension of input 0 (shape: {a_shape}) must match the first "
                        f"dimension of input 1 (shape: {b_shape}) but got: {a_shape[get_contracting_dim(a_shape)[0]]} and {b_shape[get_contracting_dim(b_shape, lhs=False)[0]]}"
                    ],
                )

            a_shape, b_shape = self.get_operand_shape_after_broadcast(a_shape, b_shape)

            self.batching_dim = {"lhs": get_batch_indices(a_shape), "rhs": get_batch_indices(b_shape)}
            self.contracting_dim = {"lhs": get_contracting_dim(a_shape), "rhs": get_contracting_dim(b_shape, False)}
            self.outputs[0].shape = get_output_shape(a_shape, b_shape)

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, "@")
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, flat_ir):
        from tripy.flat_ir.ops.dot import DotOp

        a_shape = self.inputs[0].shape
        b_shape = self.inputs[1].shape
        inputs = copy.copy(self.inputs)

        # Insert broadcast ops unconditionally.
        a_shape, b_shape = self.get_operand_shape_after_broadcast(a_shape, b_shape)
        inputs[0] = op_utils.insert_broadcast(self, flat_ir, inputs[0], a_shape)
        inputs[1] = op_utils.insert_broadcast(self, flat_ir, inputs[1], b_shape)

        flat_ir.add_op(
            self, DotOp, inputs, self.outputs, contracting_dim=self.contracting_dim, batching_dim=self.batching_dim
        )


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
        assert np.array_equal(out.numpy(), (np.ones((2,3), dtype=np.float32) @ np.ones((3,2), dtype=np.float32)))
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        MatrixMultiplication,
    )
