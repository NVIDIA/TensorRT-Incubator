from dataclasses import dataclass
from typing import Sequence, Tuple, Union

from tripy import utils
from tripy.common.types import ShapeInfo
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.ops.utils import raise_error_io_info, to_dims


@dataclass(repr=False)
class Reshape(BaseOperator):
    """
    Represents a reshape operation.
    """

    shape: Sequence[int]

    def infer_shapes(self):
        assert len(self.inputs) == 1, "Reshape operation should have exactly one input!"
        self.outputs[0].shape = to_dims(self.shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ReshapeOp

        if any(
            (dim[0].is_dynamic_dim() or dim[1].is_dynamic_dim()) for dim in zip(inputs[0].shape, to_dims(self.shape))
        ):
            raise NotImplementedError("Dynamic reshape is not supported")

        ReshapeOp(self, inputs, outputs)


@dataclass(repr=False)
class Squeeze(Reshape):
    """
    Represents a squeeze operation.
    """

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
                    raise_error_io_info(
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


@TENSOR_METHOD_REGISTRY("reshape")
def reshape(self, shape: ShapeInfo) -> "tripy.Tensor":
    """
    Returns a new tensor with the contents of this one in the specified shape.

    Args:
        shape: The desired shape.

    Returns:
        A new tensor of the same data type as this one and the specified shape.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.ones((2, 3), dtype=tp.float32)
        output = input.reshape((1, 6))

        assert np.array_equal(output.numpy(), np.reshape(np.ones((2, 3), dtype=np.float32), (1, 6)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Reshape, shape)


@TENSOR_METHOD_REGISTRY("squeeze")
def squeeze(self, dims: Union[Tuple, int] = None) -> "tripy.Tensor":
    """
    Returns a new tensor with all specified singleton dimensions of this tensor removed.

    Args:
        dims: The singleton dimensions to be removed.
              If this is not provided, all dimensions of size 1 are removed.

    Raises:
        TripyException: If any of the specified dimensions have a size that is not equal to 1.

    Returns:
        A new tensor of the same data type as this one.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.ones((1, 2, 1), dtype=tp.float32)
        # Squeeze all dimensions:
        squeeze_all = input.squeeze()
        # Squeeze only the first dimension:
        squeeze_0 = input.squeeze(0)
        # Squeeze the first and third dimensions:
        squeeze_0_2 = input.squeeze((0, 2))

        assert np.array_equal(squeeze_all.numpy(), np.squeeze(np.ones((1, 2, 1), dtype=np.float32)))
        assert np.array_equal(squeeze_0.numpy(), np.squeeze(np.ones((1, 2, 1), dtype=np.float32), 0))
        assert np.array_equal(squeeze_0_2.numpy(), np.squeeze(np.ones((1, 2, 1), dtype=np.float32), (0, 2)))
    """
    from tripy.frontend import Tensor

    if isinstance(dims, int):
        dims = utils.make_tuple(dims)

    return Tensor.build([self], Squeeze, None, dims)
