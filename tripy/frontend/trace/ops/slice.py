import math
from dataclasses import dataclass
from typing import Tuple, Union, List
from tripy import utils
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import make_tuple
from tripy.frontend.trace.ops.concatenate import Concatenate
from tripy.frontend.trace.ops.binary_elementwise import Comparison
from tripy.frontend.trace.ops.binary_elementwise import BinaryElementwise
from tripy.common.exception import raise_error


@dataclass(repr=False)
class Slice(BaseTraceOp):

    index: Tuple[Union[slice, int]]

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_shapes(self):
        input_shape = self.inputs[0].shape
        self.start_indices, self.limit_indices, self.strides = op_utils.get_slice_indices(self, input_shape, self.index)
        out_shape = [
            math.ceil((stop - start) / stride)
            for start, stop, stride in zip(self.start_indices, self.limit_indices, self.strides)
        ]
        self.outputs[0].shape = utils.to_dims(out_shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import AddOp, CompareOp, SelectOp, DynamicSliceOp
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.common.datatype import int32, bool

        device = inputs[0].device
        input_shape = op_utils.get_shape_of_tensor(inputs[0])
        input_rank = len(inputs[0].shape)

        zero_1d = op_utils.add_constant_tensor_from_list([0], device)

        def slice_rank1_tensor(rank1_tensor, slice_index):
            """
            Slice rank 1 tensor along a certain index.
            Ex: tensor [1,2,3,4,5,6] sliced at slice_index 2 will return 3.
            """
            start_idx = op_utils.add_constant_tensor_from_list([slice_index], device)
            stride_index = op_utils.add_constant_tensor_from_list([1], device)

            slice_len = op_utils.add_constant_tensor_from_list([slice_index + 1], device)
            shape_slice = FlatIRTensor.build(
                shape=utils.to_dims([1]),
                dtype=int32,
                device=device,
                reason_details=[
                    "slice rank 1 tensor ",
                    rank1_tensor,
                    f" describing slice parameters to get the slice parameter value at {slice_index} dimension.",
                ],
            )
            DynamicSliceOp.build([rank1_tensor, start_idx, slice_len, stride_index], [shape_slice])
            return shape_slice

        def convert_to_positive_idx(index_tensor, dim):
            """
            If the value of index_tensor is less than 0, this function adds the dimension value 'dim' to the index_tensor to get the real index.
            Ex: if t is of shape (2,3) and index at 0th dim is -1, then this function will add -1 with 2 to return 1.

            The below code lowers the following code:
            return index_tensor > 0 ? index_tensor : index_tensor + dim
            """
            comparison_out = FlatIRTensor.build(
                shape=utils.to_dims([1]),
                dtype=bool,
                device=device,
                reason_details=["compare if index tensor ", index_tensor, " is greater than 0."],
            )
            CompareOp.build(
                [index_tensor, zero_1d],
                [comparison_out],
                compare_direction=Comparison.Kind.GREATER_EQUAL.compare_direction,
            )

            add_out = FlatIRTensor.build(
                shape=utils.to_dims([1]),
                dtype=int32,
                device=device,
                reason_details=["add 1 to index tensor", index_tensor, " to get the real dimension value."],
            )
            AddOp.build([index_tensor, dim], [add_out])

            select_out = FlatIRTensor.build(
                shape=utils.to_dims([1]),
                dtype=int32,
                device=device,
                reason_details=["select ", index_tensor, " if ", comparison_out, " is true else select", add_out],
            )

            SelectOp.build([comparison_out, index_tensor, add_out], [select_out])

            return select_out

        start_index_tensor = inputs[1]
        limit_index_tensor = inputs[2]
        stride_index_tensor = inputs[3]

        start_slices = []
        limit_slices = []
        stride_slices = []

        def compute_slice_param_value_at_index(
            slices_tensor: List, index_tensor: FlatIRTensor, shape_slice: FlatIRTensor, idx: int, name: str
        ):

            with FlatIRTensor.context(
                [
                    f"compute the {name} value at index {idx} by converting element {idx} in ",
                    index_tensor,
                    " to a positive index.",
                ]
            ):
                slices_tensor.append(convert_to_positive_idx(slice_rank1_tensor(index_tensor, idx), shape_slice))

        for idx in range(input_rank):
            shape_slice = slice_rank1_tensor(input_shape, idx)
            compute_slice_param_value_at_index(
                start_slices,
                start_index_tensor,
                shape_slice,
                idx,
                "start slice",
            )
            compute_slice_param_value_at_index(limit_slices, limit_index_tensor, shape_slice, idx, "limit slice")
            compute_slice_param_value_at_index(stride_slices, stride_index_tensor, shape_slice, idx, "stride slice")

        start_index_tensor = op_utils.concatenate_tensors(start_slices, dim=0)
        limit_index_tensor = op_utils.concatenate_tensors(limit_slices, dim=0)
        stride_index_tensor = op_utils.concatenate_tensors(stride_slices, dim=0)
        DynamicSliceOp.build([inputs[0], start_index_tensor, limit_index_tensor, stride_index_tensor], outputs)


@TENSOR_METHOD_REGISTRY("__getitem__")
def __getitem__(self, index: Union[slice, int, Tuple[int]]) -> "tripy.Tensor":
    """
    Returns a tensor containing a slice of this tensor.

    Args:
        index: The index or slice.

    Returns:
        A tensor containing the slice of this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (1, 2, 3, 1))
        output = input[:, 1:2, :-1, 0]
        assert np.array_equal(output.numpy(), np.arange(6, dtype=np.float32).reshape((1, 2, 3, 1))[:, 1:2, :-1, 0])
    """
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.trace.ops.concatenate import concatenate
    from tripy.frontend.trace.ops.where import where
    from tripy.frontend.trace.ops.reshape import squeeze
    from tripy.frontend.trace.ops.unsqueeze import unsqueeze
    from tripy.common.datatype import int32

    index = make_tuple(index)

    # List of slices of start, limit and stride tensors.
    # These slices will be concatenated to create the start, limit and stride tensor with the same rank as input tensor.
    start_index_list, limit_index_list, stride_index_list = [], [], []

    # Initialize constant tensors which can be used multiple times in the below method.
    zero = Tensor([0], dtype=int32)
    ones = Tensor([1], dtype=int32)

    def process_int_or_tensor_element(val):
        """
        This function creates a Tensor from val where val can be an integer or a Tensor.
        """
        assert isinstance(val, int) or isinstance(val, Tensor)
        return Tensor([val], dtype=int32) if isinstance(val, int) else val

    # index can be a tuple of just integer, Tensor (ex; a[2] or a[t]) or can be a slice with optional start, stop and step fields set (where the element can be int or Tensor).
    t_shape = self.shape
    for i, idx in enumerate(index):

        if isinstance(idx, int) or isinstance(idx, Tensor):
            t_idx_corrected = process_int_or_tensor_element(idx)
            start_index_list.append(t_idx_corrected)
            # Base condition for t_shape[i] else the frontend will recurse infinitely.
            if isinstance(idx, int) and idx >= 0:
                t_idx_corrected_limit = t_idx_corrected + 1
            else:
                t_idx_corrected_limit = where(
                    t_idx_corrected >= 0, t_idx_corrected + 1, unsqueeze(t_shape[i], 0) + t_idx_corrected + 1
                )
            limit_index_list.append(t_idx_corrected_limit)
            stride_index_list.append(Tensor([1], dtype=int32))

        elif isinstance(idx, slice):

            def handle_slice_index(slice_type, index_list, default_val):
                if slice_type is not None:
                    index_list.append(process_int_or_tensor_element(slice_type))
                else:
                    index_list.append(default_val)

            handle_slice_index(idx.start, start_index_list, zero)
            handle_slice_index(idx.stop, limit_index_list, unsqueeze(t_shape[i], 0))
            handle_slice_index(idx.step, stride_index_list, ones)

        else:
            raise_error(
                "Slice index type is not supported.",
                [
                    f"Slice index (or elements within start, stop, step) can only be int or Tensor. ",
                    f"Got type={type(idx).__name__}.",
                ],
            )

    # Concatenate slices of 1d tensors to get start, limit and stride tensors.
    start_idx = concatenate(start_index_list, dim=0)
    limit_idx = concatenate(limit_index_list, dim=0)
    strides_idx = concatenate(stride_index_list, dim=0)
    out = Slice.build([self, start_idx, limit_idx, strides_idx], index)

    squeeze_dims = []
    for i, idx in enumerate(index):
        if isinstance(idx, (tuple, list)):
            raise NotImplementedError("Gather is not supported")
        if isinstance(idx, int):
            squeeze_dims.append(i)
    if squeeze_dims:
        out = squeeze(out, make_tuple(squeeze_dims))

    return out
