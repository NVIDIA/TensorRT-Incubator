from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import tripy.frontend.trace.ops.utils as op_utils
from tripy import export, utils
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Reshape(BaseTraceOp):

    shape: Sequence[int]

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_rank(self):
        if self.shape is None:
            if self.inputs[1].shape is None:
                from tripy.backend.mlir.utils import ShapeContext

                out_shape = ShapeContext().get_shape_of_dynamic_trace_tensor(self.inputs[1])
                assert len(out_shape) == 1
                assert out_shape[0] > 0, f"incorrect shape computation {out_shape}"
                self.inputs[1].shape = utils.to_dims(out_shape)
            self.outputs[0].rank = self.inputs[1].shape[0].runtime_value
        else:
            self.outputs[0].rank = len(self.shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicReshapeOp

        output_shape = (
            inputs[1] if len(inputs) == 2 else op_utils.add_constant_tensor_from_list(self.shape, inputs[0].device)
        )
        DynamicReshapeOp.build([inputs[0], output_shape], outputs)


@dataclass(repr=False)
class Squeeze(BaseTraceOp):

    dims: Tuple[int]
    out_shape: List[int]

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_rank(self):

        if len(self.dims) > 0:
            self.outputs[0].rank = self.inputs[0].rank - len(self.dims)
        else:
            from tripy.backend.mlir.utils import ShapeContext

            input_0_shape = ShapeContext().get_shape_of_dynamic_trace_tensor(self.inputs[0])

            def squeeze_shape(shape, indices_to_squeeze):
                # Convert shape to list if it's not already
                shape = list(shape)
                if not indices_to_squeeze:  # If the list is empty, squeeze all dimensions that are 1
                    shape = [dim for dim in shape if dim != 1]
                else:
                    # Sort indices to squeeze in descending order to avoid index shifting issues
                    indices_to_squeeze.sort(reverse=True)
                    for idx in indices_to_squeeze:
                        if shape[idx] == 1:
                            shape.pop(idx)
                        else:
                            raise ValueError(f"Cannot squeeze dimension at index {idx} with value {shape[idx]}")

                return shape

            out_shape = squeeze_shape(input_0_shape, list(self.dims))
            self.outputs[0].rank = len(out_shape)
            self.out_shape = out_shape

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicReshapeOp

        if len(self.dims) > 0:
            select_indices = [i for i in range(inputs[0].rank) if i not in self.dims]
            input_shape = op_utils.get_shape_of_tensor(inputs[0])
            shape_slice = []
            for index in select_indices:
                shape_slice.append(op_utils.slice_rank1_tensor(input_shape, index, reason_details=""))

            output_shape = (
                op_utils.concatenate_tensors(shape_slice, dim=0)
                if len(shape_slice) > 0
                else op_utils.add_constant_tensor_from_list([], inputs[0].device)
            )

        else:
            output_shape = op_utils.add_constant_tensor_from_list(self.out_shape, inputs[0].device)
        DynamicReshapeOp.build([inputs[0], output_shape], outputs)


@export.public_api(document_under="tensor_operations")
def reshape(input: "tripy.Tensor", shape: ShapeInfo) -> "tripy.Tensor":
    """
    Returns a new tensor with the contents of the input tensor in the specified shape.

    Args:
        input: The input tensor.
        shape: The desired compatible shape. If a shape dimension is -1, its value
        is inferred based on the other dimensions and the number of elements in the input.
        Atmost one dimension can be -1.

    Returns:
        A new tensor of the same data type as the input tensor and the specified shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.iota((2, 3), dtype=tp.float32)
        output = tp.reshape(input, (1, 6))

        assert np.array_equal(cp.from_dlpack(output).get(), np.reshape(cp.from_dlpack(input).get(), (1, 6)))
    """
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.trace.ops.concatenate import concatenate

    if isinstance(shape, Tensor):
        return Reshape.build([input, shape], None)

    def compute_reshape_shape(input_tensor, out_shape):
        # Compute the total number of elements in the original shape
        output_shape = Tensor(list(out_shape))
        total_elements = 1
        input_shape = input_tensor.shape
        for i in range(input_tensor.rank):
            total_elements *= input_shape[i]

        # Compute the product of known dimensions in the reshape shape
        known_dims_product = 1
        inferred_index = -1
        for i, dim in enumerate(out_shape):
            if dim == -1:
                inferred_index = i
            else:
                known_dims_product *= dim

        # Infer the dimension
        if inferred_index != -1:
            inferred_dim = total_elements / known_dims_product
            output_shape = concatenate(
                [output_shape[:inferred_index], reshape(inferred_dim, (1,)), output_shape[inferred_index + 1 :]], dim=0
            )

        return output_shape

    if -1 in shape:
        if shape.count(-1) > 1:
            raise_error(f"Reshape operation size operand can have only one dimension as -1, got shape={shape}.")
        shape = compute_reshape_shape(input, shape)
        return Reshape.build([input, shape], None)

    return Reshape.build([input], shape)


@export.public_api(document_under="tensor_operations")
def squeeze(input: "tripy.Tensor", dims: Union[Tuple, int] = None) -> "tripy.Tensor":
    """
    Returns a new tensor with all specified singleton dimensions of the input tensor removed.

    Args:
        input: The input tensor.
        dims: The singleton dimensions to be removed.
              If this is not provided, all dimensions of size 1 are removed.

    Raises:
        TripyException: If any of the specified dimensions have a size that is not equal to 1.

    Returns:
        A new tensor of the same data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Squeeze All Dimensions

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, dims=(0, 2))
        assert np.array_equal(cp.from_dlpack(output).get(), np.squeeze(cp.from_dlpack(input).get()))


    .. code-block:: python
        :linenos:
        :caption: Squeeze First Dimension

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, 0)
        assert np.array_equal(cp.from_dlpack(output).get(), np.squeeze(cp.from_dlpack(input).get(), 0))

    .. code-block:: python
        :linenos:
        :caption: Squeeze First And Third Dimension

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, (0, 2))

        assert np.array_equal(cp.from_dlpack(output).get(), np.squeeze(cp.from_dlpack(input).get(), (0, 2)))
    """

    if isinstance(dims, int):
        dims = utils.make_tuple(dims)
    elif dims is None:
        raise_error(f"Reshape operation size operand can have only one dimension as -1, got shape.")

    return Squeeze.build([input], dims, None)
