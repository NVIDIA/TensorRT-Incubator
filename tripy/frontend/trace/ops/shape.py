from dataclasses import dataclass

from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import Result


@dataclass(repr=False)
class Shape(BaseTraceOp):

    # always return a shape
    def infer_shape_output_idxs(self, inputs) -> Result:
        return Result.ok([0])

    def infer_rank(self):
        assert len(self.inputs) == 1, "ShapeOf operation should have exactly one input!"
        self.outputs[0].rank = 1

    def infer_dtypes(self):
        from tripy.common.datatype import int32

        self.outputs[0].dtype = int32

    def to_flat_ir(self, inputs, outputs):
        import tripy.frontend.trace.ops.utils as op_utils

        op_utils.get_shape_of_tensor(inputs[0], outputs[0])


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
    return Shape.build([self])
