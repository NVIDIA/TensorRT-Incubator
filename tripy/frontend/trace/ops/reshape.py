from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import tripy.frontend.trace.ops.utils as op_utils
from tripy import utils
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import export


@dataclass(repr=False)
class Reshape(BaseTraceOp):

    shape: Sequence[int]

    def infer_shapes(self):
        assert len(self.inputs) == 1, "Reshape operation should have exactly one input!"
        shape = self.inputs[0].shape
        input_volume = utils.volume(shape)
        reshape_volume = utils.volume(self.shape)
        if -1 in self.shape:
            neg_count = self.shape.count(-1)
            if neg_count != 1:
                raise_error("Only one dimension can be -1.", details=[f"Shape was: {self.shape}"])
            missing_dim = input_volume // -reshape_volume
            self.shape = tuple(missing_dim if dim == -1 else dim for dim in self.shape)
        self.outputs[0].shape = utils.to_dims(self.shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ReshapeOp

        if any(
            (dim[0].is_dynamic_dim() or dim[1].is_dynamic_dim())
            for dim in zip(inputs[0].shape, utils.to_dims(self.shape))
        ):
            raise NotImplementedError("Dynamic reshape is not supported.")

        ReshapeOp.build(inputs, outputs)


@dataclass(repr=False)
class Squeeze(Reshape):

    dims: Tuple[int]

    def infer_shapes(self):
        assert len(self.inputs) == 1, "Squeeze operation should have exactly one input!"
        input_shape = self.inputs[0].shape

        out_shape = []
        for idx, d in enumerate(input_shape):
            if self.dims is None:
                if d.runtime_value == 1:
                    continue
                out_shape.append(d.runtime_value)
            elif idx in self.dims:
                if d.runtime_value != 1:
                    op_utils.raise_error_io_info(
                        self,
                        "Cannot select an axis to squeeze out which has size not equal to one",
                        [
                            "Input tensor has shape: ",
                            input_shape,
                            " but trying to squeeze out dim: ",
                            idx,
                            " with size: ",
                            d.runtime_value,
                        ],
                    )
            else:
                out_shape.append(d.runtime_value)
        self.shape = out_shape

        super().infer_shapes()


@export.public_api(document_under="tensor")
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

        assert np.array_equal(output.numpy(), np.reshape(input.numpy(), (1, 6)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], Reshape, shape)


@export.public_api(document_under="tensor")
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
        output = tp.squeeze(input)
        assert np.array_equal(output.numpy(), np.squeeze(input.numpy()))


    .. code-block:: python
        :linenos:
        :caption: Squeeze First Dimension

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, 0)
        assert np.array_equal(output.numpy(), np.squeeze(input.numpy(), 0))

    .. code-block:: python
        :linenos:
        :caption: Squeeze First And Third Dimension

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, (0, 2))

        assert np.array_equal(output.numpy(), np.squeeze(input.numpy(), (0, 2)))
    """
    from tripy.frontend import Tensor

    if isinstance(dims, int):
        dims = utils.make_tuple(dims)

    return Tensor.build([input], Squeeze, None, dims)
