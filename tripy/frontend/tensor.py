from textwrap import indent
from typing import Any, List, Optional, Union

# Import ops to populate the registry before we define our Tensor class
import tripy.common.datatype
import tripy.frontend.ops
import tripy.frontend.trace.ops
from tripy import export, utils
from tripy.common.array import Array
from tripy.common.types import ShapeInfo
from tripy.common.utils import get_element_type, is_supported_array_type, convert_frontend_dtype_to_tripy_dtype
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
    # Allocate float32 and cast to unsupported tripy array types like int8.
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
        stack_info: Optional["StackInfo"] = None,
    ) -> None:
        """
        Creates a tensor.

        Args:
            data: The data with which to initialize the tensor.
            shape: The shape of the tensor.
            dtype: The data type of the tensor.
            device: The device on which to allocate the tensor.
            name: The name of the tensor. If provided, this must be a unique string.
            stack_info: The stack infomation of the tensor.

        .. code-block:: python
            :linenos:
            :caption: Example

            tensor = tp.Tensor([1.0, 2.0, 3.0], shape=(3,), dtype=tp.float32)
        """
        from tripy.frontend.trace.tensor import TraceTensor

        # We include code for everything above the `BaseTraceOp.build` function, which is called at most
        # this many stack frames above the constructor.
        STACK_DEPTH_OF_BUILD = 4
        # not using utils.default() because it always evaluates the `default` argument.
        stack_info = (
            stack_info if stack_info is not None else utils.get_stack_info(include_code_index=STACK_DEPTH_OF_BUILD)
        )

        name = name if name is not None else Tensor.get_unique_name()

        # set device now so we don't get errors due to not having a device attribute at all
        # in the case where data is None
        self.device = None

        # Note that most tensors won't have this field - generally only model input tensors.
        dynamic_shape = utils.to_dims(shape)

        self.trace_tensor = TraceTensor(name, stack_info, dynamic_shape, None, None, None, None)

        # Note: It is important that we are able to call the Tensor constructor with no arguments
        # since this is used internally.
        if data is not None:
            if not isinstance(data, Array):
                if isinstance(data, (List, tuple, bool, int, float)) and (
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
                # Disallow duplicate dtype/device when using Array to initialize a Tensor
                if shape is not None:
                    assert len(data.shape) == len(
                        shape
                    ), f"Rank provided to the initializer of Tensor (rank = {len(shape)}) does not match Array rank (rank = {len(data.shape)})."
                    for idx, (d1, d2) in enumerate(zip(utils.to_dims(shape), utils.to_dims(data.shape))):
                        if d2.is_dynamic_dim():
                            assert (
                                d1.is_dynamic_dim()
                            ), f"Array shape at index {idx} is dynamic (dim=({d2})) but the initializer shape at index {idx} is static (dim={d1})."
                assert not any([dtype, device]), "Duplicate arguments are not allowed. Use `Tensor(data)` instead."

            # Data is present now. Assign the underlying device type.
            self.device = data.device

            Storage.build_internal([], [self.trace_tensor], data)

            # Assign shapes to Trace tensor to the provided dynamic shape (if any) else default to the Array shape.
            self.trace_tensor.shape = utils.default(self._dynamic_shape, utils.to_dims(data.shape))

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
    def _dynamic_shape(self):
        return self.trace_tensor.shape

    @_dynamic_shape.setter
    def _dynamic_shape(self, new_shape):
        self.trace_tensor.shape = new_shape

    @property
    def stack_info(self):
        return self.trace_tensor.stack_info

    @stack_info.setter
    def stack_info(self, new_stack_info):
        self.trace_tensor.stack_info = new_stack_info

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
        # Data is present now. Assign the underlying device type.
        self.device = data.device

        Storage.build_internal([], [self.trace_tensor], data)
        return data

    def data(self) -> Array:
        import tripy.common.datatype
        from tripy.frontend.trace.ops.cast import cast
        from tripy.frontend.trace.ops.dequantize import dequantize

        arr = self.eval()
        if not is_supported_array_type(self.dtype):
            if self.dtype == tripy.common.datatype.float8:
                arr = dequantize(Tensor(arr), 1.0, tripy.common.datatype.float32).eval()
            else:
                arr = cast(Tensor(arr), tripy.common.datatype.float32).eval()
        return arr

    def __iter__(self):
        raise TypeError("Iterating over tensors is not supported")

    def __repr__(self) -> str:
        # The Evaluation required before accessing self.trace_tensor.producer attributes.
        arr = self.data()
        indentation = ""
        sep = ""
        if len(self.trace_tensor.producer.shape) > 1 and any(dim > 1 for dim in self.trace_tensor.producer.shape):
            indentation = " " * 4
            sep = "\n"
        return (
            f"tensor({sep}"
            f"{indent(str(arr), prefix=indentation)}, {sep}"
            f"{indent(f'dtype={self.trace_tensor.producer.dtype}, loc={self.trace_tensor.producer.device}, shape={self.trace_tensor.producer.shape}', prefix=indentation)}"
            f")"
        )

    # Since the underlying data is an Array we reuse their __dlpack__() and __dlpack_device__() methods
    def __dlpack__(self, stream: Any = None):
        return self.eval().__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self.eval().__dlpack_device__()
