import enum
from dataclasses import dataclass
from typing import Sequence, List, Union

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.ops.utils import to_dims
from tripy.utils import make_list


@dataclass
class Reduce(BaseOperator):
    """
    Represents a slice operation.
    """

    class Kind(str, enum.Enum):
        SUM = "sum"
        """Perform a reduce sum"""
        MAX = "max"
        """Perform a reduce max"""

    dim: Sequence[int]
    kind: Kind

    def infer_shapes(self):
        input_shape = self.inputs[0].shape

        if self.dim is None:
            out_shape = []
            self.dim = list(range(len(input_shape)))
        else:
            self.dim = make_list(self.dim)
            self.dim = [idx if idx >= 0 else idx + len(input_shape) for idx in self.dim]
            out_shape = []
            for idx, s in enumerate(input_shape):
                if idx not in self.dim:
                    out_shape.append(s)
        self.outputs[0].shape = to_dims(out_shape)

    def to_flat_ir(self, flat_ir):
        import numpy as np
        from tripy.flat_ir.ops import ConstantOp, ReduceOp

        init_value = 0  # for sum and max
        init_const = flat_ir.add_tensor(shape=[], dtype=self.outputs[0].dtype, device=self.outputs[0].device)
        flat_ir.add_op(self, ConstantOp, [], [init_const], data=np.array(init_value, dtype=self.outputs[0].dtype.name))
        flat_ir.add_op(
            self, ReduceOp, [self.inputs[0], init_const], self.outputs, reduce_mode=self.kind, reduce_dims=self.dim
        )


def _reduce_impl(self: "tripy.Tensor", kind: Reduce.Kind, dim: Union[int, Sequence], keepdim: bool):
    from tripy.frontend import Tensor

    out = Tensor.build([self], Reduce, dim, kind)
    if keepdim:
        raise NotImplementedError("Unsqueeze is not supported yet.")

    return out


@TENSOR_METHOD_REGISTRY("sum")
def sum(self: "tripy.Tensor", dim: Union[int, Sequence] = None, keepdim: bool = False):
    """
    Returns the sum of each row of the input tensor in the given dimension dim.
    If dim is a list of dimensions, reduce over all of them.

    Args:
        dim: the dimension or dimensions to reduce. If None, all dimensions are reduced.
        keepdim: whether to retain reduced dimensions in the output. If this is False, reduced dimensions will be squeezed.

    Returns:
        the reduced Tensor

    Example:
    ::

        a = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        out = a.sum(0)
        print(out)
        assert np.array_equal(out.numpy(), np.sum(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(self, Reduce.Kind.SUM, dim, keepdim)


@TENSOR_METHOD_REGISTRY("max")
def max(self: "tripy.Tensor", dim: Union[int, Sequence] = None, keepdim: bool = False):
    """
    Returns the maximum value of each row of the input tensor in the given dimension dim.
    If dim is a list of dimensions, reduce over all of them.

    Args:
        dim: the dimension or dimensions to reduce. If None, all dimensions are reduced.
        keepdim: whether to retain reduced dimensions in the output. If this is False, reduced dimensions will be squeezed.

    Returns:
        the reduced Tensor

    Example:
    ::

        a = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        out = a.max(0)
        print(out)
        assert np.array_equal(out.numpy(), np.max(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return _reduce_impl(self, Reduce.Kind.MAX, dim, keepdim)
