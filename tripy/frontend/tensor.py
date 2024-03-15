from textwrap import indent
from typing import Any, List, Optional, Union

# Import ops to populate the registry before we define our Tensor class
import tripy.frontend.ops
import tripy.frontend.trace.ops
from tripy import utils
from tripy.common.array import Array
from tripy.common.types import ShapeInfo
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops import Storage
from tripy.utils import export


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
        # Note: It is important that we are able to call the Tensor constructor with no arguments
        # since this is used internally by Tensor.build()

        # Note that most tensors won't have this field - generally only model input tensors.
        self._dynamic_shape = utils.to_dims(shape)
        if data is not None:
            if not isinstance(data, Array):
                data = Array(data, dtype, utils.from_dims(shape), device)
            else:
                # Internal usage only
                # Disallow duplicate shape/dtype/device when using Array to initialize a Tensor
                assert not any(
                    [shape, dtype, device]
                ), "Duplicate arguments are not allowed. Use `Tensor(data)` instead."
            self._finalize(name, [], Storage, data)

    def _finalize(self, name: Optional[str], inputs: List["Tensor"], OpType: type, *args, **kwargs) -> None:
        # It is very important that this is called from all entrypoints to creating a tensor.
        # We include logic here that needs to be applied to all tensors.
        from tripy.frontend.trace.tensor import TraceTensor

        # We include stack information from everything above `build` up to user code.
        # This lets us generate very nice error messages.
        # NOTE: If the call stack depth for this function changes, update the index here!
        STACK_DEPTH_IN_TENSOR = 3
        self.stack_info = utils.get_stack_info(include_code_index=STACK_DEPTH_IN_TENSOR)

        inp_trace_tensors = []
        for inp in inputs:
            assert len(inp.op.outputs) == 1, "Multiple outputs are not supported!"
            out = inp.op.outputs[0]
            inp_trace_tensors.append(out)

        name = utils.default(name, Tensor.get_unique_name())
        out_trace_tensors = [TraceTensor(name, self.stack_info, [], None, None, None)]

        self.op = OpType(inp_trace_tensors, out_trace_tensors, *args, **kwargs)

        # Update producer:
        self.op.outputs[0].producer = self.op

        # Update dtype
        self.op.infer_dtypes()
        self.dtype = self.op.outputs[0].dtype
        """Data type of this tensor"""

    # This function expects to receive a BaseTraceOp type (not instance!) along
    # with any extra arguments that it might need. It will then construct an instance
    # with inputs, outputs, and the extra arguments
    @staticmethod
    def build(inputs: List["Tensor"], OpType: type, *args, **kwargs) -> None:
        tensor = Tensor(None)
        tensor._finalize(None, inputs, OpType, *args, **kwargs)
        return tensor

    def __getattr__(self, name: str):
        import tripy as tp
        from tripy.common.exception import search_for_missing_attr

        look_in = [(tp, "tripy")]
        search_for_missing_attr("tripy.Tensor", name, look_in)

    @property
    def name(self):
        return self.op.outputs[0].name

    @name.setter
    def name(self, new_name):
        self.op.outputs[0].name = new_name

    def eval(self) -> Array:
        from tripy.backend.mlir.compiler import Compiler
        from tripy.backend.mlir.executor import Executor
        from tripy.backend.utils import get_tensor_info
        from tripy.frontend.trace import Trace

        if isinstance(self.op, Storage):
            return self.op.data

        trace = Trace([self])
        flat_ir = trace.to_flat_ir()
        mlir = flat_ir.to_mlir()
        compiler = Compiler(trt_builder_opt_level=0)
        executable = compiler.compile(mlir, flat_ir=flat_ir)
        executor = Executor(executable, get_tensor_info(flat_ir.outputs))
        # Upon computing the value of this tensor, we switch it to have a `Storage`
        # parameter so that it does not need to be computed again.
        data = executor.execute()
        assert len(data) == 1, "Expects only one output from mlir_tensorrt.compiler executor"
        data = data[0]
        self._finalize(self.name, [], Storage, data)
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
            f"{indent(f'dtype={self.op.dtype}, loc={self.op.device}, shape={self.op.shape}', prefix=indentation)}"
            f"{sep})"
        )

    # Since the underlying data is numpy/cupy we reuse their __dlpack__() methods
    def __dlpack__(self, stream: Any = None):
        array = self.eval().view()
        return array.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        array = self.eval().view()
        return array.__dlpack_device__()
