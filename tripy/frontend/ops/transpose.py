from dataclasses import dataclass
from typing import Sequence

from tripy import util
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


@dataclass
class Transpose(BaseOperator):
    """
    Represents a transpose operation.
    """

    permutation: Sequence[int]
    dim0: int
    dim1: int

    def to_trace_str(self, input_names, output_names):
        assert len(input_names) == 1, "Transpose operation should exactly one input!"
        assert len(output_names) == 1, "Transpose operation should have exactly one output!"
        return f"{output_names[0]} = Tensor.transpose(perm={self.permutation})"

    def infer_shapes(self, input_shapes):
        assert len(input_shapes) == 1, "Transpose operation should have exactly one input!"
        origin_shape = input_shapes[0]

        if self.permutation is None:
            # invoked via transpose()
            perm = list(range(len(origin_shape)))
            perm[self.dim0], perm[self.dim1] = perm[self.dim1], perm[self.dim0]
            self.permutation = perm

        if len(self.permutation) != len(origin_shape):
            raise Exception(
                f"Transpose.permutation must be a permutation of [0, dim(input)), got permutation={self.permutation} and dim(input)={len(origin_shape)}"
            )
        perm_shape = [origin_shape[idx] for idx in self.permutation]
        return [util.make_tuple(perm_shape)]

    def infer_dtypes(self, input_dtypes):
        assert len(input_dtypes) == 1, "Transpose operation should have exactly one input!"
        return [input_dtypes[0]]

    def infer_devices(self, input_devices):
        assert len(input_devices) == 1, "Transpose operation should have exactly one input!"
        return [input_devices[0]]

    def to_flat_ir(self, flat_ir, inputs, outputs):
        from tripy.flat_ir.ops import TransposeOp

        flat_ir.ops.append(TransposeOp(self, inputs, outputs, perm=self.permutation))


@TENSOR_METHOD_REGISTRY("transpose")
def transpose(self: "tripy.Tensor", dim0: int, dim1: int):
    """
    Returns a tensor that is a transposed version of input.
    The given dimensions dim0 and dim1 are swapped.

    Args:
        dim0: the first dimension to be transposed
        dim1: the second dimension to be transposed

    Returns:
        the transposed Tensor

    Example:
    ::

        import numpy as np

        t = np.random.rand(2, 3).astype(np.float32)
        a = tp.Tensor(t)
        out = a.transpose(0, 1)
        assert (out.numpy() == np.transpose(t, (1, 0))).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Transpose, None, dim0, dim1)


@TENSOR_METHOD_REGISTRY("permute")
def permute(self: "tripy.Tensor", perm: Sequence[int]):
    """
    Returns a tensor with its dimensions permuted.

    Args:
        perm: the desired ordering of dimensions. it must be a tuple or list
              that contains a permutation of `[0, 1, ..., N-1]`, where N is the number of dimensions of input

    Returns:
        the output Tensor with its dimensions permuted

    Example:
    ::

        import numpy as np

        t = np.random.rand(2, 3).astype(np.float32)
        a = tp.Tensor(t)
        out = a.permute((1, 0))
        assert (out.numpy() == np.transpose(t, (1, 0))).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Transpose, perm, None, None)
