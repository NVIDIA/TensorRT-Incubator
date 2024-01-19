import functools
from dataclasses import dataclass
from typing import List

from tripy import utils
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.dim import Dim
from tripy.utils import make_list, make_tuple


# TODO: Move this class elsewhere since we'll probably want to use it all over the place.
@dataclass
class ConditionCheck:
    """
    Bundles a boolean with error message details.

    This can be used in cases where we would like to perform a check in some helper, e.g.
    `is_broadcast_compatible` but still display a nice error message with low level feedback
    from said helper (in this example, that could be details on which dimensions are not compatible).

    The caller can access the `details` field for more information on the error message and
    provide it to `raise_error`.
    """

    value: bool
    details: List[str]

    def __bool__(self) -> bool:
        return self.value


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


def get_slice_indices(shape, index):
    """
    Converts index to slices required by Slice operation

    Args:
        shape: shape of input tensor
        index: tuple of slices or int

    Returns:
        start_indices: list of start slice index
        limit_indices: list of end slice index
        strides: list of slice strides
    """
    # TODO: only works for static shape, figure out how to handle DS
    runtime_shape = [dim.runtime_value for dim in shape]
    dims = len(shape)
    if len(index) > dims:
        raise_error(
            "Too many indices for array.",
            details=[
                "Array has dim of ",
                dims,
                " but was indexed with ",
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
            start_indices.append(to_positive_idx(idx.start, dim) if idx.start else 0)
            limit_indices.append(to_positive_idx(idx.stop, dim) if idx.stop else dim)
            strides.append(idx.step if idx.step else 1)
    return start_indices, limit_indices, strides


def get_broadcast_compatible_shapes(shape1, shape2):
    # Make the shorter shape the same length as the longer shape by padding with ones
    if len(shape1) > len(shape2):
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2
    elif len(shape2) > len(shape1):
        shape1 = (1,) * (len(shape2) - len(shape1)) + shape1

    return to_dims(shape1), to_dims(shape2)


def is_broadcast_compatible(shape1, shape2) -> ConditionCheck:
    # Now check each dimension pair
    for index, (dim1, dim2) in enumerate(zip(shape1, shape2)):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return ConditionCheck(
                False,
                [
                    f"for tensor shapes: {shape1} and {shape2}, dimensions on axis: {index} ({dim1} and {dim2}) are not broadcast compatible."
                ],
            )

    return ConditionCheck(True, [])


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


# Decorator to preprocess inputs of a function and convert numpy, python types to tripy tensors.
def allow_non_tensor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from tripy.frontend.tensor import Tensor

        # Only convert args to tripy tensor. kwargs are allowed to be of non-tripy tensor type.
        new_args = [Tensor(arg) if not isinstance(arg, Tensor) else arg for arg in args]
        return func(*new_args, **kwargs)

    return wrapper


# To which dimension in the target shape each dimension of the operand shape corresponds to.
def get_broadcast_in_dim(input_shape, output_shape):
    broadcast_dimensions = []
    rank_diff = len(output_shape) - len(input_shape)

    for idx, dim in enumerate(input_shape):
        corresponding_output_dim = idx + rank_diff

        # We might need careful check in case of dynamic dims
        broadcast_dimensions.append(corresponding_output_dim)

    return broadcast_dimensions
