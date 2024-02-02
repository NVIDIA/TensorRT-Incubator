import functools
from typing import List, Union

from tripy import utils
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.dim import Dim
from tripy.utils import make_list, make_tuple


def to_dims(shape: ShapeInfo):
    """
    Convert the given shape tuple to a tuple of Dim objects.

    Args:
        shape (Tuple[Union[int, Dim]]): The input shape.

    Returns:
        Tuple[Dim]: The converted shape as a tuple of Dim objects.
    """
    if shape is None:
        return None

    dims = make_list(shape)
    for i in range(len(dims)):
        if not isinstance(dims[i], Dim):
            dims[i] = Dim(dims[i])
    return make_tuple(dims)


# Like raise_error but adds information about the inputs and output.
def raise_error_io_info(op, summary, details) -> None:
    assert len(op.outputs) == 1, "This helper should only be used for ops with a single output!"
    details = ["For expression:", op.outputs[0]] + details + ["\n\n"]
    for index, inp in enumerate(op.inputs):
        details.extend([f"Input {index} was:", inp])

    raise_error(summary, details)


def _check_input_attr_matches(
    op: "BaseOperator", op_details: str, attr: str, attr_name: str, start_index: int = None, stop_index: int = None
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
                "].",
            ],
        )


# Checks whether properties of the inputs match. Optional index parameters can be provided in case not all inputs should be considered.
def check_input_dtypes_match(op: "BaseOperator", op_details: str = "", start_index: int = None, stop_index: int = None):
    return _check_input_attr_matches(op, op_details, "dtype", "data type", start_index, stop_index)


def check_input_shapes_match(op: "BaseOperator", op_details: str = "", start_index: int = None, stop_index: int = None):
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
## Broadcasting
##


def get_broadcast_compatible_shapes(shape1, shape2):
    # Make the shorter shape the same length as the longer shape by padding with ones
    if len(shape1) > len(shape2):
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2
    elif len(shape2) > len(shape1):
        shape1 = (1,) * (len(shape2) - len(shape1)) + shape1

    return to_dims(shape1), to_dims(shape2)


def is_broadcast_compatible(shape1, shape2) -> utils.ConditionCheck:
    # Now check each dimension pair
    for index, (dim1, dim2) in enumerate(zip(shape1, shape2)):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return utils.ConditionCheck(
                False,
                [
                    f"for tensor shapes: {shape1} and {shape2}, dimensions on axis: {index} ({dim1} and {dim2}) are not broadcast compatible."
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

    return broadcast_dimensions


# Insert a broadcast op into the flat_ir which broadcasts input tensor to output shape.
# If the output shape is dynamic, shape of the target_tensor is used to describe the output shape.
def insert_broadcast(
    origin_layer: "BaseOperator",
    input_tensor: "FIRTensor",
    out_shape: ShapeInfo,
    use_dynamic_variant: bool = False,
    target_tensor: "FIRTensor" = None,
):
    from tripy.flat_ir.ops import BroadcastOp, DynamicBroadcastOp, ShapeOp
    from tripy.flat_ir.tensor import FIRTensor
    from tripy.frontend.ops.utils import get_broadcast_in_dim

    output_tensor = FIRTensor.build(shape=out_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    if use_dynamic_variant:
        from tripy import int32

        assert target_tensor, "target_tensor is required for dynamic variant of the broadcast op."

        # insert a shape tensor
        shape_output_tensor = FIRTensor.build(
            shape=(Dim(len(target_tensor.shape)),), dtype=int32, device=input_tensor.device
        )

        ShapeOp(origin_layer, [target_tensor], [shape_output_tensor])

        DynamicBroadcastOp(
            origin_layer,
            [input_tensor, shape_output_tensor],
            [output_tensor],
            broadcast_dim=get_broadcast_in_dim(input_tensor.shape, out_shape),
        )

    else:
        BroadcastOp(
            origin_layer,
            [input_tensor],
            [output_tensor],
            broadcast_dim=get_broadcast_in_dim(input_tensor.shape, out_shape),
        )
    return output_tensor
