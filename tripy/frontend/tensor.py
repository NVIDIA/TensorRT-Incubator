from textwrap import indent
from typing import Any, List, Optional

from tripy import utils
from tripy.common.array import Array
from tripy.common.logging import logger
from tripy.frontend.ops import TENSOR_METHOD_REGISTRY, Storage


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


class Tensor(metaclass=TensorMeta):
    """
    A tensor is a multi-dimensional array that contains elements of a uniform data type.
    """

    _COUNT = 0

    @classmethod
    def get_unique_name(cls):
        name = f"t{cls._COUNT}"
        cls._COUNT += 1
        return name

    # This field communicates to NumPy that it should allow our right-side operator overloads (e.g. __radd__) to take
    # precedence over its own left-side overloads (e.g. __add__). This will ensure that an expression of the form
    # `<np_array> <binary_op> Tensor` will return a Tensor and not a NumPy array.
    __array_priority__ = 10000

    def _finalize(self, name: Optional[str], inputs: List["Tensor"], OpType: type, *args, **kwargs) -> None:
        # It is very important that this is called from all entrypoints to creating a tensor.
        # We include logic here that needs to be applied to all tensors.
        from tripy.frontend.trace.tensor import TraceTensor

        # We include stack information from everything above `build` up to user code.
        # This lets us generate very nice error messages.
        # NOTE: If the call stack depth for this function changes, update the index here!
        STACK_DEPTH_IN_TENSOR = 3
        self._stack_info = utils.get_stack_info(include_code_index=STACK_DEPTH_IN_TENSOR)

        inp_trace_tensors = []
        for inp in inputs:
            assert len(inp.op.outputs) == 1, "Multiple outputs are not supported!"
            out = inp.op.outputs[0]
            inp_trace_tensors.append(out)

        name = utils.default(name, Tensor.get_unique_name())
        out_trace_tensors = [TraceTensor(name, self._stack_info, [], None, None, None)]

        self.op = OpType(inp_trace_tensors, out_trace_tensors, *args, **kwargs)

        # Update producer:
        self.op.outputs[0].producer = self.op

    # This function expects to receive a BaseOperator type (not instance!) along
    # with any extra arguments that it might need. It will then construct an instance
    # with inputs, outputs, and the extra arguments
    @staticmethod
    def build(inputs: List["Tensor"], OpType: type, *args, **kwargs) -> None:
        tensor = Tensor(None)
        tensor._finalize(None, inputs, OpType, *args, **kwargs)
        return tensor

    @property
    def name(self):
        return self.op.outputs[0].name

    @name.setter
    def name(self, new_name):
        self.op.outputs[0].name = new_name

    def eval(self) -> Array:
        from tripy.backend.jit.utils import get_tensor_info
        from tripy.backend.mlir.compiler import FlatIRCompiler
        from tripy.backend.mlir.executor import FlatIRExecutor
        from tripy.frontend.trace import Trace

        if isinstance(self.op, Storage):
            return self.op.data

        trace = Trace([self])
        flat_ir = trace.to_flat_ir()
        compiler = FlatIRCompiler()
        executable = compiler.compile(flat_ir)
        with FlatIRExecutor(executable, get_tensor_info(flat_ir.inputs), get_tensor_info(flat_ir.outputs)) as executor:
            # Upon computing the value of this tensor, we switch it to have a `Storage`
            # parameter so that it does not need to be computed again.
            data = executor.execute()
            assert len(data) == 1, "Expects only one output from MLIR executor"
            data = data[0]
            # TODO (#114): Remove shape argument
            self._finalize(self.name, [], Storage, data, data.shape)
            return data

    def numpy(self) -> "numpy.ndarray":
        from tripy.common.device import device

        # TODO (#114): Insert a self.eval() here so we don't need to recompute everything
        # again after calling `numpy()`
        # self.eval()
        data = self.to(device("cpu")).eval()
        # TODO (#114): Remove this line after adding self.eval()
        self._finalize(self.name, [], Storage, data, data.shape)
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
