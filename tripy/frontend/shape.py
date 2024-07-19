from tripy import export, utils
from typing import Optional, Sequence, Union
from tripy.common.array import Array
from tripy.common.exception import raise_error
from tripy.frontend.tensor import Tensor
from tripy.common.device import device
from tripy.common.datatype import int32


@export.public_api()
class Shape(Tensor):
    """
    A Shape is a tensor used to represent a tensor shape.
    Shapes are vectors (rank 1) of non-negative integers (using int32 as the datatype).
    """

    # set to higher than tensor's so adding a shape to a tensor will use shape's overload (and give an error)
    __array_priority__ = 20000

    def __init__(
        self,
        data: Union[Sequence, Tensor, Array, "np.ndarray", "cp.ndarray", "torch.Tensor", "jnp.ndarray"],
        num_dims: Optional[Union[int, "tripy.dynamic_dim"]] = None,
        name: Optional[str] = None,
    ) -> None:
        r"""
        Create a new shape.

        Args:
            data: The value of the shape, which should be a 1D array of integers (the dimensions).
            num_dims: The number of dimensions in the shape (its rank), which should correspond to the number of elements in data
            name: An optional name
        """

        from tripy.common.exception import raise_error

        if isinstance(data, Tensor):
            # these fields can be None in the case of an uninitialized tensor (like Tensor(None))
            if data.trace_tensor.rank is not None and data.trace_tensor.rank != 1:
                raise_error(f"Shape tensors must be of rank 1, but input tensor is rank {data.rank}", details=[data])
            if data.dtype is not None and data.dtype != int32:
                raise_error(
                    f"Shape tensors must have int32 members, but input tensor has data type {data.dtype}",
                    details=[data],
                )

            # the shape of data should correspond to the given rank
            super().__init__(data=None, shape=utils.to_dims(num_dims), dtype=int32, name=name, device=data.device)
            # the device field is not set by the superclass constructor if the data field is not passed
            self.device = data.device
            # share the underlying data
            self.trace_tensor = data.trace_tensor
            self.stack_info = data.stack_info
        else:
            shape = data.shape if hasattr(data, "shape") else utils.get_shape(data)
            device = data.device if hasattr(data, "device") else None
            if len(shape) != 1:
                raise_error(
                    f"Tensors used to represent shapes must be of rank 1, but given shape {shape} has rank {len(shape)}."
                )
            # for an array, duplicate fields are not allowed
            if isinstance(data, Array):
                super().__init__(data=data, shape=utils.to_dims(num_dims))
            else:
                super().__init__(data=data, shape=utils.to_dims(num_dims), dtype=int32, name=name, device=device)

    def as_tensor(self) -> Tensor:
        """
        Return an ordinary Tripy `Tensor` with the same contents as this `Shape`. No copying is done.

        Returns:
            A `Tensor` with the same underlying value as the current `Shape`.

        .. code-block:: python
            :linenos:
            :caption: Example

            s = tp.Shape([1, 2, 3])
            t = s.as_tensor()
            assert isinstance(t, tp.Tensor) and not isinstance(t, tp.Shape)
            assert np.array_equal(cp.from_dlpack(s).get(), cp.from_dlpack(t).get())
        """
        ret = Tensor(data=None, shape=utils.to_dims(self.rank), dtype=int32, name=self.name, device=self.device)
        ret.trace_tensor = self.trace_tensor
        ret.stack_info = self.stack_info
        return ret

    def add(self, other: Union["tp.Shape", Tensor]) -> "tp.Shape":
        """
        The + operator for shapes is concatenation. This method is exposed to allow for elementwise addition,
        should it be necessary.

        Args:
            other: Another Shape or Tensor (though the result will be a Shape).

        Returns:
            The result of elementwise addition of this `Shape` and `other`.

        .. code-block:: python
            :linenos:
            :caption: Example

            s1 = tp.Shape([1, 2, 3])
            s2 = tp.Shape([4, 5, 6])
            res = s1.add(s2)
            assert isinstance(res, tp.Shape)
            assert cp.from_dlpack(res).get().tolist() == [5, 7, 9]
        """
        return super().__add__(other)

    def __repr__(self) -> str:
        # denote the representation as a shape rather than a tensor
        tensor_repr = super().__repr__()
        assert tensor_repr[:6] == "tensor"
        return "shape" + tensor_repr[6:]

    # addition for shapes is concatenation, not tensor addition

    def __add__(self, other):
        from tripy.frontend.trace.ops.concatenate import concatenate

        if not isinstance(other, Shape) and isinstance(other, Tensor):
            raise_error(
                "Attempting to add a Tripy Tensor to a Tripy Shape, which is not allowed. Consider calling tp.Shape explicitly"
            )
        elif not isinstance(other, Shape):
            other = Shape(other)
        return concatenate([self, other], 0)

    def __radd__(self, other):
        from tripy.frontend.trace.ops.concatenate import concatenate

        if not isinstance(other, Shape) and isinstance(other, Tensor):
            raise_error(
                "Attempting to add a Tripy Tensor to a Tripy Shape, which is not allowed. Consider calling tp.Shape explicitly"
            )
        elif not isinstance(other, Shape):
            other = Shape(other)
        return concatenate([other, self], 0)