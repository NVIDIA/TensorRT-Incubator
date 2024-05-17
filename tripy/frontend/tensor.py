from textwrap import indent
from typing import Any, List, Optional, Union

# Import ops to populate the registry before we define our Tensor class
import tripy.common.datatype
import tripy.frontend.ops
import tripy.frontend.trace.ops
import tripy.third_party.utils
from tripy import export, utils
from tripy.common.array import Array
from tripy.common.types import ShapeInfo
from tripy.common.utils import is_supported_array_type, get_element_type
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops import Storage


class TensorMeta(type):
    def __new__(cls, name, bases, dct):
        new = type.__new__(cls, name, bases, dct)

        # We only register methods with the Tensor class. Derived classes
        # will inherit these methods normally. If we register for derived classes too
        # we run the risk of overwriting overridden methods.
        if name == "Tensor":
            # Add methods specified by individual ops to this class.
            for method_name in TENSOR_METHOD_REGISTRY:
                setattr(new, method_name, TENSOR_METHOD_REGISTRY[method_name])

        return new


def convert_list_data_to_array(data, shape, dtype, device):
    from tripy.frontend.trace.ops.cast import cast
    from tripy.frontend.trace.ops.quantize import quantize

    # (183) Initialize Array with an arbitrary data type. Requires implementing "create_memref_and_cast" API.
    if dtype == tripy.common.datatype.float8:
        return quantize(
            Tensor(Array(data, tripy.common.datatype.float32, utils.from_dims(shape), device)), 1.0, dtype
        ).eval()
    return cast(Tensor(Array(data, tripy.common.datatype.float32, utils.from_dims(shape), device)), dtype).eval()


@export.public_api(
    document_under="tensor/index.rst",
    autodoc_options=[
        ":special-members:",
        ":exclude-members: __init__, __repr__, __weakref__, __dlpack__, __dlpack_device__",
    ],
)
class Tensor(metaclass=TensorMeta):
    """
    A tensor is a multi-dimensional array that contains elements of a uniform data type.
    """

    _COUNT = 0

    # This field communicates to NumPy that it should allow our right-side operator overloads (e.g. __radd__) to take
    # precedence over its own left-side overloads (e.g. __add__). This will ensure that an expression of the form
    # `<np_array> <binary_op> Tensor` will return a Tensor and not a NumPy array.
    __array_priority__ = 10000

    @classmethod
    def get_unique_name(cls):
        name = f"t{cls._COUNT}"
        cls._COUNT += 1
        return name

    def __init__(
        self,
        data: Union[List, "np.ndarray", "cp.ndarray", "torch.Tensor", "jnp.ndarray"],
        shape: Optional[ShapeInfo] = None,
        dtype: Optional["tripy.dtype"] = None,
        device: Optional["tripy.device"] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Creates a tensor.

        Args:
            data: The data with which to initialize the tensor.
            shape: The shape of the tensor.
            dtype: The data type of the tensor.
            device: The device on which to allocate the tensor.
            name: The name of the tensor. If provided, this must be a unique string.

        .. code-block:: python
            :linenos:
            :caption: Example

            tensor = tp.Tensor([1.0, 2.0, 3.0], shape=(3,), dtype=tp.float32)
        """
        from tripy.frontend.trace.tensor import TraceTensor

        self.stack_info = utils.get_stack_info()

        name = utils.default(name, Tensor.get_unique_name())
        self.trace_tensor = TraceTensor(name, self.stack_info, [], None, None, None, None)

        # Note that most tensors won't have this field - generally only model input tensors.
        self._dynamic_shape = utils.to_dims(shape)

        # Note: It is important that we are able to call the Tensor constructor with no arguments
        # since this is used internally.
        if data is not None:
            if not isinstance(data, Array):
                if isinstance(data, (int, float, List, tuple)) and (
                    (not is_supported_array_type(dtype))
                    or get_element_type(data) == tripy.common.datatype.float32
                    and dtype == tripy.common.datatype.int32
                ):
                    # 1. Allocate float32 and cast to unsupported types.
                    # 2. Allocage float32 and cast to int32 to be compliant with numpy/cupy behavior.
                    data = convert_list_data_to_array(data, shape, dtype, device)
                else:
                    data = Array(data, dtype, utils.from_dims(shape), device)
            else:
                # Internal usage only
                # Disallow duplicate shape/dtype/device when using Array to initialize a Tensor
                assert not any(
                    [shape, dtype, device]
                ), "Duplicate arguments are not allowed. Use `Tensor(data)` instead."
            Storage.build_internal([], [self.trace_tensor], data)

    def __getattr__(self, name: str):
        import tripy as tp
        from tripy.common.exception import search_for_missing_attr

        look_in = [(tp, "tripy")]
        search_for_missing_attr("tripy.Tensor", name, look_in)

    @property
    def name(self):
        return self.trace_tensor.name

    @name.setter
    def name(self, new_name):
        self.trace_tensor.name = new_name

    @property
    def dtype(self):
        return self.trace_tensor.dtype

    @property
    def rank(self):
        return self.trace_tensor.rank

    def eval(self) -> Array:
        from tripy.backend.mlir.compiler import Compiler
        from tripy.backend.mlir.executor import Executor
        from tripy.backend.utils import get_devices, get_tensor_info
        from tripy.frontend.trace import Trace

        if isinstance(self.trace_tensor.producer, Storage):
            return self.trace_tensor.producer.data

        trace = Trace([self])
        flat_ir = trace.to_flat_ir()
        mlir = flat_ir.to_mlir()

        compiler = Compiler(trt_builder_opt_level=0)
        executable = compiler.compile(mlir, flat_ir=flat_ir)
        output_tensor_info = get_tensor_info(flat_ir.outputs)
        executor = Executor(executable)
        # Upon computing the value of this tensor, we switch it to have a `Storage`
        # parameter so that it does not need to be computed again.
        data = executor.execute(get_devices(output_tensor_info))
        assert len(data) == 1, "Expects only one output from mlir_tensorrt.compiler executor"
        data = data[0]
        Storage.build_internal([], [self.trace_tensor], data)
        return data

    def data(self) -> List[Union[float, int]]:
        import tripy.common.datatype
        from tripy.frontend.trace.ops.cast import cast
        from tripy.frontend.trace.ops.dequantize import dequantize

        if not is_supported_array_type(self.dtype):
            if self.dtype == tripy.common.datatype.float8:
                data = dequantize(self, 1.0, tripy.common.datatype.float32).eval()
            else:
                data = cast(self, tripy.common.datatype.float32).eval()
        else:
            data = self.eval()  # Avoid recomputing everything after we've called `numpy()`
        assert isinstance(data, Array)
        return data

    def __repr__(self) -> str:
        arr = self.data()
        assert isinstance(arr, Array)
        indentation = ""
        sep = ""
        if len(arr.shape) > 1 and any(dim > 1 for dim in arr.shape):
            indentation = " " * 4
            sep = "\n"
        return (
            f"tensor({sep}"
            f"{indent(str(arr), prefix=indentation)}, {sep}"
            f"{indent(f'dtype={arr.dtype}, loc={arr.device}, shape={arr.shape}', prefix=indentation)}"
            f"{sep})"
        )

    # Since the underlying data is an Array we reuse their __dlpack__() and __dlpack_device__() methods
    def __dlpack__(self, stream: Any = None):
        return self.eval().__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self.eval().__dlpack_device__()
