from typing import List

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
        return max(dim1, dim2)


##
## Helpers
##


def get_shape_of_tensor(tensor: "FlatIRTensor"):
    from tripy.common.array import Array
    from tripy.common.datatype import int32
    from tripy.flat_ir.ops import ConstantOp, ShapeOp
    from tripy.flat_ir.tensor import FlatIRTensor

    shape_output_tensor = FlatIRTensor.build(
        shape=(dynamic_dim(len(tensor.shape)),),
        dtype=int32,
        device=tensor.device,
        reason_details=["retrieve the shape of: ", tensor],
    )
    if len(tensor.shape) > 0:
        ShapeOp.build([tensor], [shape_output_tensor])
    else:
        # TODO #80: Remove this codepath when shape dialect is used (shape.shape_of).
        ConstantOp.build(
            [],
            [shape_output_tensor],
            data=Array(None, int32, shape=(0,), device=tensor.device),
        )
    return shape_output_tensor


def add_constant_tensor_from_list(op: "BaseTraceOp", data: list, device: "tripy.device"):
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.flat_ir.ops import ConstantOp
    from tripy.common.datatype import int32
    import numpy as np

    const_output_tensor = FlatIRTensor.build(
        shape=(dynamic_dim(1),),
        dtype=int32,
        device=device,
        reason_details=[f"create constant rank 1 int32 tensor filled with {data}."],
    )
    ConstantOp.build([], [const_output_tensor], data=np.array(data).astype(np.int32))
    return const_output_tensor


def concatenate_tensors(op: "BaseTraceOp", inputs: List["FlatIRTensor"], dim: int):
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.flat_ir.ops import ConcatenateOp
    from tripy.common.datatype import int32

    out = FlatIRTensor.build(
        shape=utils.to_dims(
            -1,
        ),
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
# tensor_details should describe what this tensor is (e.g. left operand of '+')
def insert_broadcast(
    input_tensor: "FlatIRTensor",
    out_shape: ShapeInfo,
    tensor_details: str,
    use_dynamic_variant: bool = False,
    shape_of_target_tensor: "FlatIRTensor" = None,
):
    from tripy.flat_ir.ops import BroadcastOp, DynamicBroadcastOp
    from tripy.flat_ir.tensor import FlatIRTensor
    from tripy.frontend.trace.ops.utils import get_broadcast_in_dim

    output_tensor = FlatIRTensor.build(
        shape=out_shape,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
        reason_details=[
            f"broadcast the {tensor_details}, which was: ",
            input_tensor,
            f"to a shape of: {out_shape} in order to be compatible with the other input(s)",
        ],
    )

    if use_dynamic_variant:
        from tripy import int32

        assert shape_of_target_tensor, "shape_of_target_tensor is required for dynamic variant of the broadcast op."

        DynamicBroadcastOp.build(
            [input_tensor, shape_of_target_tensor],
            [output_tensor],
            broadcast_dim=get_broadcast_in_dim(input_tensor.shape, out_shape),
        )

    else:
        BroadcastOp.build(
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
    assert nb_extra_dims > 0
    const_val_tensor = FlatIRTensor.build(
        shape=[], dtype=int32, device=input.device, reason_details=f"create a rank 0 constant tensor filled with 1."
    )
    ones_shape_tensor = FlatIRTensor.build(
        shape=utils.to_dims(
            nb_extra_dims,
        ),
        dtype=int32,
        device=input.device,
        reason_details=[f"create a rank 1 shape tensor filled {nb_extra_dims} ones."],
    )

    ConstantOp.build([], [const_val_tensor], data=np.array(1, dtype=np.int32))
    BroadcastOp.build([const_val_tensor], [ones_shape_tensor], broadcast_dim=[])

    shape_of_input = get_shape_of_tensor(input)
    concat_output_tensor = FlatIRTensor.build(
        shape=utils.to_dims(
            nb_extra_dims + len(input.shape),
        ),
        dtype=int32,
        device=input.device,
        reason_details=[
            f"append {nb_extra_dims} ones to the input shape {shape_of_input} to expand the rank of tensor."
        ],
    )
    ConcatenateOp.build([ones_shape_tensor, shape_of_input], [concat_output_tensor], dim=0)

    # output shape usage just relies on rank.
    output_shape = utils.to_dims((1,) * nb_extra_dims + input.shape)

    return insert_broadcast(
        input, output_shape, use_dynamic_variant=True, shape_of_target_tensor=concat_output_tensor, tensor_details=""
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
