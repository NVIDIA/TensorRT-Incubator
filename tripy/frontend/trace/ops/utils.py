from typing import Any, List, Optional

from colored import Fore, attr

from tripy import utils
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.dim import dynamic_dim
from tripy.utils import Result


# Like raise_error but adds information about the inputs and output.
def raise_error_io_info(op, summary, details) -> None:
    assert len(op.outputs) == 1, "This helper should only be used for ops with a single output!"
    details += [":"] + [op.outputs[0]]
    for index, inp in enumerate(op.inputs):
        details.extend([f"{Fore.magenta}Input {index} was:{attr('reset')}", inp])

    raise_error(summary, details)


def _check_input_attr_matches(
    op: "BaseTraceOp", op_details: str, attr: str, attr_name: str, start_index: int = None, stop_index: int = None
):
    assert len(op.inputs), "This function must not be called for operations without inputs!"

    start = utils.default(start_index, 0)
    stop = utils.default(stop_index, len(op.inputs))

    inp_range_str = "all inputs"
    if start_index is not None or stop_index is not None:
        inp_range_str = f"inputs [{start}-{stop - 1}]"

    assert start < len(op.inputs), "Start index cannot be larger than number of inputs!"

    inputs = op.inputs[start:stop]

    if any(getattr(inp, attr) != getattr(inputs[0], attr) for inp in inputs):
        dtypes = []
        for index, inp in enumerate(inputs):
            dtypes.extend([", " if index > 0 else "", getattr(inp, attr)])

        raise_error_io_info(
            op,
            f"Incompatible input {attr_name}s.",
            details=[
                f"For operation: '{op_details}', " if op_details else "For this operation, ",
                f"{attr_name}s for {inp_range_str}" " must match, but got: [",
                *dtypes,
                "]",
            ],
        )


# Checks whether properties of the inputs match. Optional index parameters can be provided in case not all inputs should be considered.
def check_input_dtypes_match(op: "BaseTraceOp", op_details: str = "", start_index: int = None, stop_index: int = None):
    return _check_input_attr_matches(op, op_details, "dtype", "data type", start_index, stop_index)


def check_input_shapes_match(op: "BaseTraceOp", op_details: str = "", start_index: int = None, stop_index: int = None):
    return _check_input_attr_matches(op, op_details, "shape", "shape", start_index, stop_index)


def get_broadcast_dim(dim1, dim2):
    if dim1.is_dynamic_dim():
        return dim1
    elif dim2.is_dynamic_dim():
        return dim2
    else:
        assert dim1 == 1 or dim2 == 1 or dim1 == dim2
        # can't just return max(dim1, dim2) because one may be 0
        if dim1 == 1:
            return dim2
        # dim1 == dim2 or dim2 == 1
        return dim1


##
## Helpers
##


def get_shape_of_tensor(tensor: "FlatIRTensor"):
    from tripy.common.array import Array
    from tripy.common.datatype import int32
    from tripy.flat_ir.ops import ConstantOp, ShapeOp
    from tripy.flat_ir.tensor import FlatIRTensor

    shape_output_tensor = FlatIRTensor.build(
        rank=1,
        dtype=int32,
        device=tensor.device,
        reason_details=["retrieve the shape of: ", tensor],
    )
    if tensor.rank > 0:
        ShapeOp.build([tensor], [shape_output_tensor])
    else:
        # TODO #80: Remove this codepath when shape dialect is used (shape.shape_of).
        ConstantOp.build(
            [],
            [shape_output_tensor],
            data=Array(None, int32, shape=(0,), device=tensor.device),
        )
    return shape_output_tensor


def add_constant_tensor_from_list(data: list, device: "tripy.device"):
    from tripy.common.array import Array
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.flat_ir.ops import ConstantOp
    from tripy.common.datatype import int32
    from tripy.common.device import device

    const_output_tensor = FlatIRTensor.build(
        shape=utils.to_dims((len(data),)),
        rank=1,
        dtype=int32,
        device=device,
        reason_details=[f"create constant rank 1 int32 tensor filled with {data}."],
    )
    ConstantOp.build(
        [],
        [const_output_tensor],
        data=Array(data, shape=[len(data)], dtype=int32, device=device("cpu")),
    )
    return const_output_tensor


def concatenate_tensors(inputs: List["FlatIRTensor"], dim: int):
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.flat_ir.ops import ConcatenateOp
    from tripy.common.datatype import int32

    out = FlatIRTensor.build(
        rank=1,
        dtype=int32,
        device=inputs[0].device,
        reason_details=[
            "output of concatenation of the following tensors: ",
            *[inp for inp in inputs],
            f" along dim {dim}.",
        ],
    )
    ConcatenateOp.build(inputs, [out], dim=dim)
    return out


##
## Broadcasting
##


def get_broadcast_compatible_shapes(shape1, shape2):
    # Make the shorter shape the same length as the longer shape by padding with ones
    if len(shape1) > len(shape2):
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2
    elif len(shape2) > len(shape1):
        shape1 = (1,) * (len(shape2) - len(shape1)) + shape1

    return utils.to_dims(shape1), utils.to_dims(shape2)


def is_broadcast_compatible(shape1, shape2) -> Result:
    # Now check each dimension pair
    for index, (dim1, dim2) in enumerate(zip(shape1, shape2)):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return Result.err(
                [
                    f"for tensor shapes: {shape1} and {shape2}, dimensions on axis {index}: '{dim1}' and '{dim2}' are not broadcast compatible"
                ],
            )

    return Result.ok()


# Given two shapes, compute the shape of the resulting broadcast. Assumes that the shapes are of equal rank
def compute_shape_of_broadcast(
    shape1, shape2, output_rank: int, shape1_name: Optional[str] = None, shape2_name: Optional[str] = None
):
    from tripy.common.datatype import int32, bool as tp_bool
    from tripy.flat_ir.ops import CompareOp, SelectOp
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.frontend.trace.ops.binary_elementwise import Comparison

    shape1_name = utils.default(shape1_name, "a tensor")
    shape2_name = utils.default(shape2_name, "another tensor")

    # can't just use the max of shape1 and shape2 because it will be incorrect if a dim is 0
    # (the broadcast of 0 and 1 is 0)
    resulting_shape = FlatIRTensor.build(
        shape=utils.to_dims([output_rank]),
        rank=1,
        dtype=int32,
        device=shape1.device,
        reason_details=[
            f"compute the broadcasted shape of {shape1_name} ",
            shape1,
            f" and {shape2_name} ",
            shape2,
        ],
    )
    shape_dim_comparison = FlatIRTensor.build(
        shape=utils.to_dims([output_rank]),
        rank=1,
        dtype=tp_bool,
        device=shape1.device,
        reason_details=[
            f"Compare the dims of {shape1_name} with 1",
        ],
    )
    ones = add_constant_tensor_from_list([1] * output_rank, shape1.device)
    # if shape1[i] == 1, use shape2[i]. Otherwise use shape1[i]
    CompareOp.build([shape1, ones], [shape_dim_comparison], compare_direction=Comparison.Kind.EQUAL.compare_direction)
    SelectOp.build([shape_dim_comparison, shape2, shape1], [resulting_shape])
    return resulting_shape


# To which dimension in the target shape each dimension of the operand shape corresponds to.
def get_broadcast_in_dim(input_rank: int, output_rank: int) -> List[int]:
    assert output_rank >= input_rank
    broadcast_dimensions = []
    rank_diff = output_rank - input_rank

    for idx in range(input_rank):
        corresponding_output_dim = idx + rank_diff

        # We might need careful check in case of dynamic dims
        broadcast_dimensions.append(corresponding_output_dim)

    assert len(broadcast_dimensions) == input_rank
    return broadcast_dimensions


# Insert a broadcast op into the flat_ir which broadcasts input tensor to output shape.
# If the output shape is dynamic, shape_of_target_tensr is used to describe the output shape.
# tensor_details should describe what this tensor is (e.g. left operand of '+')
def insert_broadcast(
    input_tensor: "FlatIRTensor",
    out_rank: int,
    tensor_details: str,
    use_dynamic_variant: bool = False,
    shape_of_target_tensor: "FlatIRTensor" = None,
):
    from tripy.flat_ir.ops import BroadcastOp, DynamicBroadcastOp
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.frontend.trace.ops.utils import get_broadcast_in_dim

    output_tensor = FlatIRTensor.build(
        rank=out_rank,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
        reason_details=[
            f"broadcast the {tensor_details}, which was: ",
            input_tensor,
            f" to a rank of: {out_rank} in order to be compatible with the other input(s)",
        ],
    )

    if use_dynamic_variant:

        assert shape_of_target_tensor, "shape_of_target_tensor is required for dynamic variant of the broadcast op."

        DynamicBroadcastOp.build(
            [input_tensor, shape_of_target_tensor],
            [output_tensor],
            broadcast_dim=get_broadcast_in_dim(input_tensor.rank, out_rank),
        )

    else:
        BroadcastOp.build(
            [input_tensor],
            [output_tensor],
            broadcast_dim=get_broadcast_in_dim(input_tensor.rank, out_rank),
        )
    return output_tensor


# Expands rank of a tensor via prepending extra dims provided by nb_extra_dims.
def expand_rank_of_tensor(input: "FlatIRTensor", nb_extra_dims: int):
    from tripy.common.array import Array
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.flat_ir.ops import BroadcastOp, ConcatenateOp, ConstantOp, DynamicBroadcastOp
    from tripy.common.datatype import int32
    from tripy.common.device import device
    from tripy import utils

    if nb_extra_dims == 0:
        return input

    # Create array filled with 1s and concat with shape array
    assert nb_extra_dims > 0
    # rank 1 tensor
    shape_of_input = get_shape_of_tensor(input)

    # create rank 1 tensor filled with nb_extra_dims
    extra_ones = add_constant_tensor_from_list([1] * nb_extra_dims, input.device)

    concat_output_tensor = FlatIRTensor.build(
        rank=1,
        dtype=int32,
        device=input.device,
        reason_details=[
            f"append {nb_extra_dims} ones to the input shape {shape_of_input} to expand the rank of tensor."
        ],
    )
    ConcatenateOp.build([extra_ones, shape_of_input], [concat_output_tensor], dim=0)

    # output shape usage just relies on rank.
    output_rank = input.rank + nb_extra_dims
    return insert_broadcast(
        input,
        out_rank=output_rank,
        use_dynamic_variant=True,
        shape_of_target_tensor=concat_output_tensor,
        tensor_details="",
    )


##
## Slice
##


def get_slice_indices(op, shape, index):
    """
    Converts index to slices required by Slice operation

    Args:
        shape: shape of input tensor

    Returns:
        start_indices: list of start slice index
        limit_indices: list of end slice index
        strides: list of slice strides
    """
    # TODO: only works for static shape, figure out how to handle DS
    runtime_shape = [dim.runtime_value for dim in shape]

    dims = len(shape)
    if len(index) > dims:
        raise_error_io_info(
            op,
            "Too many indices for input tensor.",
            details=[
                "Input tensor has a rank of ",
                dims,
                " but was attempted to be sliced with ",
                len(index),
                " indices",
            ],
        )
    index += (dims - len(index)) * (slice(None),)
    start_indices = []
    limit_indices = []
    strides = []
    to_positive_idx = lambda idx, dim: idx + dim if idx < 0 else idx
    for idx, dim in zip(index, runtime_shape):
        if isinstance(idx, int):
            # slice the single element and squeeze later
            idx = to_positive_idx(idx, dim)
            start_indices.append(idx)
            limit_indices.append(idx + 1)
            strides.append(1)
        else:
            if idx.step is not None and idx.step < 0:
                start_indices.append(
                    0 if idx.start is None or idx.start >= dim else (dim - to_positive_idx(idx.start, dim) - 1)
                )
                limit_indices.append(dim if idx.stop is None else (dim - to_positive_idx(idx.stop, dim) - 1))
            else:
                start_indices.append(to_positive_idx(utils.default(idx.start, 0), dim))
                # clamp the limit index if it goes past the end
                limit_indices.append(min(dim, to_positive_idx(idx.stop, dim)) if (idx.stop is not None) else dim)
            strides.append(abs(utils.default(idx.step, 1)))
    return start_indices, limit_indices, strides


def slice_rank1_tensor(rank1_tensor: "FlatIRTensor", slice_index: int, reason_details: Optional[List[Any]] = None):
    """
    Slice a rank 1 tensor tensor along a certain index.
    Ex: tensor [1,2,3,4,5,6] sliced at slice_index 2 will return 3.
    """
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.flat_ir.ops import DynamicSliceOp
    from tripy.common.datatype import int32

    device = rank1_tensor.device
    start_idx = add_constant_tensor_from_list([slice_index], device)
    stride_index = add_constant_tensor_from_list([1], device)
    slice_len = add_constant_tensor_from_list([slice_index + 1], device)
    result_slice = FlatIRTensor.build(
        rank=1,
        dtype=int32,
        device=device,
        reason_details=reason_details if reason_details is not None else [],
    )
    DynamicSliceOp.build([rank1_tensor, start_idx, slice_len, stride_index], [result_slice])
    return result_slice
