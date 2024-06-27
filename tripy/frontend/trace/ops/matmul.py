from dataclasses import dataclass

from tripy import utils
import tripy.frontend.trace.ops.utils as op_utils
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class MatrixMultiplication(BaseTraceOp):
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

    def infer_rank(self):
        if self.inputs[0].rank == 1 and self.inputs[1].rank == 1:
            self.outputs[0].rank = 0
        else:
            self.outputs[0].rank = max(inp.rank for inp in self.inputs)

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
                        f"inputs for operation: '@' must have at least one dimension, but input {index} has shape: {shape} which has fewer than 1 dimension"
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
                        f"for operation: '@', the second dimension of input 0 (shape: {a_shape}) must match the first "
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

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.flat_ir.ops import ConcatenateOp, DotOp, DynamicSliceOp, MaxOp
        from tripy.common.datatype import int32

        # Following steps are followed in the implementation below:
        # 1. Slice the input shape into batch dims and matrix dims.
        # 2. Ensure that batch dims for both operands are of same rank by prepending shape tensor with 1s.
        # 3. Apply broadcasting rule to batch dims to get the output batch dims.
        # 4. Concatenate the batch dims with matrix dims computed in step1.
        # 5. Use the computed output dims from #4 to broadcast both the inputs.
        # 6. Invocate DotOp.

        def split_shape_in_batch_and_mat_dims(input: "FlatIRTensor", nb_batch_dims: int):
            input_shape = op_utils.get_shape_of_tensor(input)

            zero_1d = op_utils.add_constant_tensor_from_list([0], input.device)
            one_1d = op_utils.add_constant_tensor_from_list([1], input.device)

            slice_len = op_utils.add_constant_tensor_from_list([nb_batch_dims], input.device)
            batch_slice = FlatIRTensor.build(
                shape=utils.to_dims([nb_batch_dims]),
                rank=1,
                dtype=int32,
                device=input.device,
                reason_details=["slice the input shape ", input_shape, " to get batch dims."],
            )
            DynamicSliceOp.build([input_shape, zero_1d, slice_len, one_1d], [batch_slice])

            end_len = op_utils.add_constant_tensor_from_list([len(input.shape)], input.device)
            mat_slice = FlatIRTensor.build(
                shape=utils.to_dims([len(input.shape) - nb_batch_dims]),
                rank=1,
                dtype=int32,
                device=input.device,
                reason_details=["slice the input shape ", input_shape, " into mat dims."],
            )
            DynamicSliceOp.build([input_shape, slice_len, end_len, one_1d], [mat_slice])
            return batch_slice, mat_slice

        def append_ones_data_tensor(input, nb_ones):
            extra_a_ones = op_utils.add_constant_tensor_from_list([1] * nb_ones, input.device)
            input_expanded = FlatIRTensor.build(
                shape=utils.to_dims(-1),
                rank=1,
                dtype=int32,
                device=input.device,
                reason_details=[f"append {nb_ones} ones to the shape tensor ", input],
            )
            ConcatenateOp.build([extra_a_ones, input], [input_expanded], dim=0)
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

        # Apply broadcasting rule of batch shapes to get the resulting batch shapes
        bcast_of_batch_shapes = op_utils.compute_shape_of_broadcast(
            a_batch_shapes_with_ones,
            b_batch_shapes_with_ones,
            nb_result_batch_dims,
            shape1_name="the batch dims of a",
            shape2_name="the batch dims of b",
        )

        # Concatenate the batch dims with matrix dims computed in step1.
        a_dims = op_utils.concatenate_tensors([bcast_of_batch_shapes, a_mat_shape], dim=0)
        b_dims = op_utils.concatenate_tensors([bcast_of_batch_shapes, b_mat_shape], dim=0)

        # Use the computed output dims from #4 to broadcast both the inputs.
        inputs[0] = op_utils.insert_broadcast(
            inputs[0],
            out_shape=utils.to_dims([-1] * (nb_result_batch_dims + a_rank - nb_a_batch_dims)),
            out_rank=nb_result_batch_dims + a_rank - nb_a_batch_dims,
            use_dynamic_variant=True,
            shape_of_target_tensor=a_dims,
            tensor_details=["left operand of DotOp"],
        )
        inputs[1] = op_utils.insert_broadcast(
            inputs[1],
            out_shape=utils.to_dims([-1] * (nb_result_batch_dims + b_rank - nb_b_batch_dims)),
            out_rank=nb_result_batch_dims + b_rank - nb_b_batch_dims,
            use_dynamic_variant=True,
            shape_of_target_tensor=b_dims,
            tensor_details=["right operand of DotOp"],
        )

        DotOp.build(inputs, outputs, contracting_dim=self.contracting_dim, batching_dim=self.batching_dim)


@TENSOR_METHOD_REGISTRY("__matmul__")
def __matmul__(self, other: "tripy.Tensor") -> "tripy.Tensor":
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

        a = tp.iota((2, 3), dtype=tp.float32)
        b = tp.iota((3, 2), dtype=tp.float32)

        output = a @ b
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(a).get() @ cp.from_dlpack(b).get())
    """
    return MatrixMultiplication.build([self, other])
