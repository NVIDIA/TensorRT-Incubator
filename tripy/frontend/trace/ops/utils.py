from colored import Fore, attr

from tripy import utils
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.dim import Dim


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
    if dim1.is_static_dim() and dim2.is_static_dim():
        assert dim1 == 1 or dim2 == 1 or dim1 == dim2
        return max(dim1, dim2)
    else:
        if dim1.is_dynamic_dim():
            return dim1
        else:
            return dim2


##
## Helpers
##


def get_shape_of_tensor(op: "BaseTraceOp", tensor: "FlatIRTensor"):
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.flat_ir.ops import ShapeOp, ConstantOp
    from tripy.common.datatype import int32

    shape_output_tensor = FlatIRTensor.build(shape=(Dim(len(tensor.shape)),), dtype=int32, device=tensor.device)
    if len(tensor.shape) > 0:
        ShapeOp(op, [tensor], [shape_output_tensor])
    else:
        # TODO #80: Remove this codepath when shape dialect is used (shape.shape_of).
        ConstantOp(op, [], [shape_output_tensor], data=b"")

    return shape_output_tensor


def add_constant_tensor_from_list(op: "BaseTraceOp", data: list, device: "tripy.device"):
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.flat_ir.ops import ConstantOp
    from tripy.common.datatype import int32
    import numpy as np

    const_output_tensor = FlatIRTensor.build(shape=(Dim(1),), dtype=int32, device=device)
    ConstantOp(op, [], [const_output_tensor], data=np.array(data).astype(np.int32))
    return const_output_tensor


# Returns the element wise maximum of shape of two tensors. This routine is used to get output shapes when dealing with broadcast.
def get_max_of_shapes(op: "BaseTraceOp", input1: "FlatIRTensor", input2: "FlatIRTensor") -> "FlatIRTensor":
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.common.datatype import int32
    from tripy.flat_ir.ops import MaxOp

    max_output_shape_tensor = FlatIRTensor.build(shape=input1.shape, dtype=int32, device=input1.device)
    MaxOp(op, [get_shape_of_tensor(op, input1), get_shape_of_tensor(op, input2)], [max_output_shape_tensor])
    return max_output_shape_tensor


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


def is_broadcast_compatible(shape1, shape2) -> utils.ConditionCheck:
    # Now check each dimension pair
    for index, (dim1, dim2) in enumerate(zip(shape1, shape2)):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return utils.ConditionCheck(
                False,
                [
                    f"for tensor shapes: {shape1} and {shape2}, dimensions on axis {index}: '{dim1}' and '{dim2}' are not broadcast compatible"
                ],
            )

    return utils.ConditionCheck(True, [])


# To which dimension in the target shape each dimension of the operand shape corresponds to.
def get_broadcast_in_dim(input_shape, output_shape):
    broadcast_dimensions = []
    rank_diff = len(output_shape) - len(input_shape)

    for idx, _ in enumerate(input_shape):
        corresponding_output_dim = idx + rank_diff

        # We might need careful check in case of dynamic dims
        broadcast_dimensions.append(corresponding_output_dim)

    assert len(broadcast_dimensions) == len(input_shape)
    return broadcast_dimensions


# Insert a broadcast op into the flat_ir which broadcasts input tensor to output shape.
# If the output shape is dynamic, shape_of_target_tensr is used to describe the output shape.
def insert_broadcast(
    source_op: "BaseTraceOp",
    input_tensor: "FlatIRTensor",
    out_shape: ShapeInfo,
    use_dynamic_variant: bool = False,
    shape_of_target_tensor: "FlatIRTensor" = None,
):
    from tripy.flat_ir.ops import BroadcastOp, DynamicBroadcastOp, ShapeOp
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.frontend.trace.ops.utils import get_broadcast_in_dim

    output_tensor = FlatIRTensor.build(shape=out_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    if use_dynamic_variant:
        from tripy import int32

        assert shape_of_target_tensor, "target_tensor is required for dynamic variant of the broadcast op."

        # get_shape_of_tensor(source_op, target_tensor)

        DynamicBroadcastOp(
            source_op,
            [input_tensor, shape_of_target_tensor],
            [output_tensor],
            broadcast_dim=get_broadcast_in_dim(input_tensor.shape, out_shape),
        )

    else:
        BroadcastOp(
            source_op,
            [input_tensor],
            [output_tensor],
            broadcast_dim=get_broadcast_in_dim(input_tensor.shape, out_shape),
        )
    return output_tensor


# Expands rank of a tensor via prepending extra dims provided by nb_extra_dims.
def expand_rank_of_tensor(op: "BaseTraceOp", input: "FlatIRTensor", nb_extra_dims: int):
    import numpy as np
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.flat_ir.ops import BroadcastOp, ConcatenateOp, ConstantOp
    from tripy.common.datatype import int32
    from tripy import utils

    if nb_extra_dims == 0:
        return input

    # Create array filled with 1s and concat with shape array
    const_val_tensor = FlatIRTensor.build(shape=[], dtype=int32, device=input.device)
    ones_shape_tensor = FlatIRTensor.build(
        shape=utils.to_dims(
            nb_extra_dims,
        ),
        dtype=int32,
        device=input.device,
    )

    ConstantOp(op, [], [const_val_tensor], data=np.array(1, dtype=np.int32))
    BroadcastOp(op, [const_val_tensor], [ones_shape_tensor], broadcast_dim=[])

    concat_output_tensor = FlatIRTensor.build(
        shape=utils.to_dims(
            nb_extra_dims + len(input.shape),
        ),
        dtype=int32,
        device=input.device,
    )

    shape_of_input = get_shape_of_tensor(op, input)
    ConcatenateOp(op, [ones_shape_tensor, shape_of_input], [concat_output_tensor], dim=0)

    # output shape usage just relies on rank.
    output_shape = utils.to_dims((1,) * nb_extra_dims + input.shape)

    return insert_broadcast(
        op, input, output_shape, use_dynamic_variant=True, shape_of_target_tensor=concat_output_tensor
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
            start_indices.append(to_positive_idx(idx.start, dim) if (idx.start is not None) else 0)
            limit_indices.append(to_positive_idx(idx.stop, dim) if (idx.stop is not None) else dim)
            strides.append(idx.step if idx.step else 1)
    return start_indices, limit_indices, strides
