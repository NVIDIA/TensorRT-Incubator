from textwrap import indent
from typing import Any, List, Optional, Union

# Import ops to populate the registry before we define our Tensor class
import tripy.frontend.ops
import tripy.frontend.trace.ops
from tripy import export, utils
from tripy.backend.mlir.utils import parse_tensor_names_from_location, redirect_stderr, remove_constants
from tripy.common.array import Array
from tripy.common.types import ShapeInfo
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
        self.trace_tensor = TraceTensor(name, self.stack_info, [], None, None, None)

        # Note that most tensors won't have this field - generally only model input tensors.
        self._dynamic_shape = utils.to_dims(shape)

        # Note: It is important that we are able to call the Tensor constructor with no arguments
        # since this is used internally.
        if data is not None:
            if not isinstance(data, Array):
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

    def eval(self) -> Array:
        from tripy.backend.mlir.compiler import Compiler
        from tripy.backend.mlir.executor import Executor
        from tripy.backend.utils import get_devices, get_runtime_shapes, get_tensor_info
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
        data = executor.execute(get_runtime_shapes(output_tensor_info), get_devices(output_tensor_info))
        assert len(data) == 1, "Expects only one output from mlir_tensorrt.compiler executor"
        data = data[0]
        Storage.build_internal([], [self.trace_tensor], data)
        return data

    def numpy(self) -> "numpy.ndarray":
        from tripy.common.device import device
        from tripy.frontend.trace.ops.copy import copy

        self.eval()  # Avoid recomputing everything after we've called `numpy()`
        data = copy(self, device("cpu")).eval()
        return data.view()

    def __repr__(self) -> str:
        np_arr = self.eval().view()
        indentation = ""
        sep = ""
        if len(np_arr.shape) > 1 and any(dim > 1 for dim in np_arr.shape):
            indentation = " " * 4
            sep = "\n"
        return (
            f"tensor({sep}"
            f"{indent(str(np_arr), prefix=indentation)}, {sep}"
            f"{indent(f'dtype={self.trace_tensor.producer.dtype}, loc={self.trace_tensor.producer.device}, shape={self.trace_tensor.producer.shape}', prefix=indentation)}"
            f"{sep})"
        )

    # Since the underlying data is numpy/cupy we reuse their __dlpack__() methods
    def __dlpack__(self, stream: Any = None):
        array = self.eval().view()
        return array.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        array = self.eval().view()
        return array.__dlpack_device__()
