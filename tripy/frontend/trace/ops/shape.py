from dataclasses import dataclass

from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Shape(BaseTraceOp):

    def infer_rank(self):
        assert len(self.inputs) == 1, "ShapeOf operation should have exactly one input!"
        self.outputs[0].rank = 1

    def infer_dtypes(self):
        from tripy import int32

        self.outputs[0].dtype = int32

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ShapeOp

        ShapeOp.build(inputs, outputs)


@TENSOR_METHOD_REGISTRY("shape")
@property
def shape(self) -> "tripy.Tensor":
    """
    Represents the shape of the tensor.

    Returns:
        A 1D tensor containing the shape of this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.ones((8, 2))
        shape = input.shape

        assert np.array_equal(cp.from_dlpack(shape).get(), np.array([8, 2]))
    """
    from tripy.frontend.tensor import Tensor
    import tripy.common.datatype

    if self.rank == 0:
        return Tensor([], dtype=tripy.common.datatype.int32)
    return Shape.build([self])
