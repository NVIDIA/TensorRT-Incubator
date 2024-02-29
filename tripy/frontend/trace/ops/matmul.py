from dataclasses import dataclass

from tripy import utils
import tripy.frontend.trace.ops.utils as op_utils
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class MatrixMultiplication(BaseTraceOp):
    """
    Represents a matrix multiplication operation.
    """

    def __str__(self):
        return f"{self.outputs[0].name} = {' @ '.join([inp.name for inp in self.inputs])}"

    def infer_shapes(self):
        # Note that infer_shapes won't reason about equality of dimensions since shape analysis is done within
        # the mlir tensorrt compiler.

        assert len(self.inputs) == 2, f"MatrixMultiplication expects exactly two inputs."
        a_rank = len(self.inputs[0].shape)
        b_rank = len(self.inputs[1].shape)

        for index, rank in enumerate([a_rank, b_rank]):
            if rank < 1:
                op_utils.raise_error_io_info(
                    self,
                    "Input tensors must have at least 1 dimension.",
                    details=[
                        f"Inputs for operation: '@' must have at least one dimension, but input {index} has rank: {rank}"
                    ],
                )

        if a_rank == 1 and b_rank == 1:
            # case 1: both operands are 1-D
            op_utils.check_input_shapes_match(self, "@")
            self.batching_dim = {"lhs": [], "rhs": []}
            self.contracting_dim = {"lhs": [0], "rhs": [0]}
            self.outputs[0].shape = tuple()
        else:
            # stablehlo dot_general requires same number of batching dims for lhs, rhs.

            def get_contracting_dim(rank, lhs=True):
                if lhs or rank == 1:
                    return [rank - 1]
                else:
                    return [rank - 2]

            def get_batch_indices(rank):
                return list(range(rank - 2))

            output_rank = max(a_rank, b_rank)
            self.batching_dim = {"lhs": get_batch_indices(output_rank), "rhs": get_batch_indices(output_rank)}
            self.contracting_dim = {
                "lhs": get_contracting_dim(output_rank),
                "rhs": get_contracting_dim(output_rank, False),
            }
            self.outputs[0].shape = utils.to_dims([-1] * (max(a_rank, b_rank)))

    def infer_dtypes(self):
        op_utils.check_input_dtypes_match(self, "@")
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.flat_ir.ops import ConcatenateOp, DotOp, DynamicSliceOp, MaxOp
        from tripy.common.datatype import int32

        # Following steps are followed in the implementation below:
        # 1. Slice the input shape into batch dims and matrix dims.
        # 2. Ensure that batch dims for both operands are of same rank by prepending shape tensor with 1s.
        # 3. Use Max of batch dims to get the output batch dims.
        # 4. Concatenate the batch dims with matrix dims computed in step1.
        # 5. Use the computed output dims from #4 to broadcast both the inputs.
        # 6. Invocate DotOp.

        def split_shape_in_batch_and_mat_dims(input: "FlatIRTensor", nb_batch_dims: int):
            input_shape = op_utils.get_shape_of_tensor(self, input)

            zero_1d = op_utils.add_constant_tensor_from_list(self, [0], input.device)
            one_1d = op_utils.add_constant_tensor_from_list(self, [1], input.device)

            slice_len = op_utils.add_constant_tensor_from_list(self, [nb_batch_dims], input.device)
            batch_slice = FlatIRTensor.build(shape=utils.to_dims([nb_batch_dims]), dtype=int32, device=input.device)
            DynamicSliceOp(self, [input_shape, zero_1d, slice_len, one_1d], [batch_slice])

            end_len = op_utils.add_constant_tensor_from_list(self, [len(input.shape)], input.device)
            mat_slice = FlatIRTensor.build(
                shape=utils.to_dims([len(input.shape) - nb_batch_dims]), dtype=int32, device=input.device
            )
            DynamicSliceOp(self, [input_shape, slice_len, end_len, one_1d], [mat_slice])
            return batch_slice, mat_slice

        def append_ones_data_tensor(input, nb_ones):
            extra_a_ones = op_utils.add_constant_tensor_from_list(self, [1] * nb_ones, input.device)
            input_expanded = FlatIRTensor.build(
                shape=utils.to_dims(-1),
                dtype=int32,
                device=input.device,
            )
            ConcatenateOp(self, [extra_a_ones, input], [input_expanded], dim=0)
            return input_expanded

        a_rank, b_rank = (len(input.shape) for input in inputs)
        nb_a_batch_dims, nb_b_batch_dims = [max(rank - 2, 0) for rank in [a_rank, b_rank]]
        nb_result_batch_dims = max(nb_a_batch_dims, nb_b_batch_dims)

        # Slice the input shape into batch dims and matrix dims.
        a_batch_shape, a_mat_shape = split_shape_in_batch_and_mat_dims(inputs[0], nb_a_batch_dims)
        b_batch_shape, b_mat_shape = split_shape_in_batch_and_mat_dims(inputs[1], nb_b_batch_dims)

        # Ensure that batch dims for both operands are of same rank by prepending shape tensor with 1s.
        a_batch_shapes_with_ones = append_ones_data_tensor(a_batch_shape, nb_result_batch_dims - nb_a_batch_dims)
        b_batch_shapes_with_ones = append_ones_data_tensor(b_batch_shape, nb_result_batch_dims - nb_b_batch_dims)

        # Use Max of batch dims to get the output batch dims.
        max_of_batch_shapes = FlatIRTensor.build(
            shape=utils.to_dims([nb_result_batch_dims]), dtype=int32, device=inputs[0].device
        )
        MaxOp(
            self,
            [a_batch_shapes_with_ones, b_batch_shapes_with_ones],
            [max_of_batch_shapes],
        )

        # Concatenate the batch dims with matrix dims computed in step1.
        a_dims = op_utils.concatenate_tensors(self, [max_of_batch_shapes, a_mat_shape], dim=0)
        b_dims = op_utils.concatenate_tensors(self, [max_of_batch_shapes, b_mat_shape], dim=0)

        # Use the computed output dims from #4 to broadcast both the inputs.
        inputs[0] = op_utils.insert_broadcast(
            self,
            inputs[0],
            utils.to_dims([-1] * (nb_result_batch_dims + a_rank - nb_a_batch_dims)),
            use_dynamic_variant=True,
            shape_of_target_tensor=a_dims,
        )
        inputs[1] = op_utils.insert_broadcast(
            self,
            inputs[1],
            utils.to_dims([-1] * (nb_result_batch_dims + b_rank - nb_b_batch_dims)),
            use_dynamic_variant=True,
            shape_of_target_tensor=b_dims,
        )

        DotOp(self, inputs, outputs, contracting_dim=self.contracting_dim, batching_dim=self.batching_dim)


@TENSOR_METHOD_REGISTRY("__matmul__")
def matmul(self, other: "tripy.Tensor") -> "tripy.Tensor":
    """
        Performs matrix multiplication between two tensors.

    - If both tensors are 1D, a dot product is performed.
        - If both tensors are 2D, matrix multiplication is performed.
        - If either argument, but not both, is 1D, a dimension is inserted
          and matrix multiplication is performed with relevant broadcast of dimension.

        Args:
            other: The tensor by which to multiply. Must have the same data type as this tensor.

        Returns:
            A new tensor of the same data type as this one.

        .. code-block:: python
            :linenos:
            :caption: Example

            a = tp.ones((2, 3), dtype=tp.float32)
            b = tp.ones((3, 2), dtype=tp.float32)

            output = a @ b

            assert np.array_equal(output.numpy(), (np.ones((2,3), dtype=np.float32) @ np.ones((3,2), dtype=np.float32)))
    """
    from tripy.frontend import Tensor

    return Tensor.build(
        [self, other],
        MatrixMultiplication,
    )
