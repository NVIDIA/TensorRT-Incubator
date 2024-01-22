from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.binary_elementwise import BinaryElementwise
from tripy.frontend.ops.copy import Copy
from tripy.frontend.ops.fill import Fill, full, full_like
from tripy.frontend.ops.iota import Iota, iota, iota_like
from tripy.frontend.ops.matmul import MatrixMultiplication
from tripy.frontend.ops.permute import Permute, Transpose, permute, transpose
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.ops.reshape import Reshape
from tripy.frontend.ops.shape import ShapeOf
from tripy.frontend.ops.slice import Slice
from tripy.frontend.ops.storage import Storage
from tripy.frontend.ops.where import Where, where
from tripy.frontend.ops.reduce import Reduce
from tripy.frontend.ops.unary_elementwise import UnaryElementwise, exp, tanh
