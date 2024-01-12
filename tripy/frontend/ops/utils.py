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
    for i in range(len(shape)):
        if not isinstance(shape[i], Dim):
            dims[i] = Dim(shape[i])
    return make_tuple(dims)


def is_broadcast_compatible(shape1, shape2) -> ConditionCheck:
    # TODO: ranks dont need to be same, implicit broadcast should be inserted to expand ranks.
    # Rank check already happens in binary elementwise which is why this is just an assertion
    assert len(shape1) == len(shape2)

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


def get_broadcast_dim(dim1, dim2):
    if dim1.is_static_dim() and dim2.is_static_dim():
        assert dim1 == 1 or dim2 == 1 or dim1 == dim2
        return max(dim1, dim2)
    else:
        from tripy.frontend.dim import Dim

        if dim1.is_dynamic_dim():
            return dim1
        else:
            return dim2


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
