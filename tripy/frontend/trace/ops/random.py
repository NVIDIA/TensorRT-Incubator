import numbers
from dataclasses import dataclass

from tripy import utils
from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Random(BaseTraceOp):
    """
    Represents a random operation.
    """

    shape: ShapeInfo
    dtype: datatype

    def infer_shapes(self):
        self.outputs[0].shape = self.shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def to_flat_ir(self, inputs, outputs, RngOp, param_a, param_b):
        RngOp.build(inputs, outputs, param_a, param_b)


@dataclass(repr=False)
class RandomUniform(Random):
    """
    Represents a random uniform operation.
    """

    low: numbers.Number
    high: numbers.Number

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import RandomUniformOp

        super().to_flat_ir(inputs, outputs, RandomUniformOp, self.low, self.high)


@dataclass(repr=False)
class RandomNormal(Random):
    """
    Represents a random normal operation.
    """

    mean: numbers.Number
    std: numbers.Number

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import RandomNormalOp

        super().to_flat_ir(inputs, outputs, RandomNormalOp, self.mean, self.std)


def rand(
    shape: ShapeInfo, low: numbers.Number = 0.0, high: numbers.Number = 1.0, dtype: datatype.dtype = datatype.float32
) -> "tripy.Tensor":
    """
    Creates a Tensor filled with random numbers from a uniform distribution on the interval [``low``, ``high``).

    Setting a random seed for reproducibility will be supported in the future with TensorRT 10.x.

    Args:
        shape: The desired shape of the tensor.
        low: Lower boundary of the output interval.
        high: Upper boundary of the output interval.
        dtype: Datatype of elements. Can only be ``tripy.float32`` or ``tripy.float16``.

    Returns:
        A tensor of shape ``shape`` with elements sampled from a uniform distribution.

    .. code-block:: python
        :linenos:
        :caption: Example

        output = tp.rand((2, 3))
    """
    from tripy.frontend import Tensor

    if dtype not in (datatype.float32, datatype.float16):
        raise_error("rand only supports float32 or float16.", [f"Got dtype={dtype}"])

    return Tensor.build([], RandomUniform, utils.to_dims(shape), dtype, low, high)


def randn(
    shape: ShapeInfo, mean: numbers.Number = 0.0, std: numbers.Number = 1.0, dtype: datatype.dtype = datatype.float32
) -> "tripy.Tensor":
    """
    Creates a Tensor filled with random numbers from a normal distribution.

    Setting a random seed for reproducibility will be supported in the future with TensorRT 10.x.

    Args:
        shape: The desired shape of the tensor.
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        dtype: Datatype of elements. Can only be ``tripy.float32`` or ``tripy.float16``.

    Returns:
        A tensor of shape ``shape`` with elements sampled from a normal distribution.

    .. code-block:: python
        :linenos:
        :caption: Example

        output = tp.randn((2, 3))
    """
    from tripy.frontend import Tensor

    if dtype not in (datatype.float32, datatype.float16):
        raise_error("randn only supports float32 or float16.", [f"Got dtype={dtype}"])

    return Tensor.build([], RandomNormal, utils.to_dims(shape), dtype, mean, std)
