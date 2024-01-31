from textwrap import indent
from typing import Any, List

from tripy import utils
from tripy.common.array import Array
from tripy.common.logging import G_LOGGER
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

    # This field communicates to NumPy that it should allow our right-side operator overloads (e.g. __radd__) to take
    # precedence over its own left-side overloads (e.g. __add__). This will ensure that an expression of the form
    # `<np_array> <binary_op> Tensor` will return a Tensor and not a NumPy array.
    __array_priority__ = 10000

    def _finalize(self, inputs: List["Tensor"], OpType: type, *args, **kwargs) -> None:
        # It is very important that this is called from all entrypoints to creating a tensor.
        # We include logic here that needs to be applied to all tensors.
        from tripy.frontend.trace.tensor import TraceTensor

        def get_name():
            name = f"t{Tensor._COUNT}"
            Tensor._COUNT += 1
            return name

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

        out_trace_tensors = [TraceTensor(get_name(), self._stack_info, [], None, None, None)]

        self.op = OpType(inp_trace_tensors, out_trace_tensors, True, *args, **kwargs)

        # Update producer:
        self.op.outputs[0].producer = self.op

    # This function expects to receive a BaseOperator type (not instance!) along
    # with any extra arguments that it might need. It will then construct an instance
    # with inputs, outputs, and the extra arguments
    @staticmethod
    def build(inputs: List["Tensor"], OpType: type, *args, **kwargs) -> None:
        tensor = Tensor(None)
        tensor._finalize(inputs, OpType, *args, **kwargs)
        return tensor

    def eval(self) -> Array:
        from tripy.backend.jit.utils import get_tensor_info
        from tripy.backend.mlir.compiler import FlatIRCompiler
        from tripy.backend.mlir.executor import FlatIRExecutor
        from tripy.frontend.trace import Trace

        if isinstance(self.op, Storage):
            return self.op.data

        trace = Trace([self])
        G_LOGGER.ir_printer(f"Trace :\n{trace}")
        flat_ir = trace.to_flat_ir()
        G_LOGGER.ir_printer(f"FlatIR :\n{flat_ir}")
        compiler = FlatIRCompiler()
        executable = compiler.compile(flat_ir)
        with FlatIRExecutor(executable, get_tensor_info(flat_ir.inputs), get_tensor_info(flat_ir.outputs)) as executor:
            # Upon computing the value of this tensor, we switch it to have a `Storage`
            # parameter so that it does not need to be computed again.
            storage_arr = executor.execute()
            assert len(storage_arr) == 1, "Expects only one output from MLIR executor"
            self.op = storage_arr[0]
            return self.op.data

    def numpy(self):
        from tripy.common.device import device

        data = self.to(device("cpu")).eval().view()
        self._finalize([], Storage, data)
        return data

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

    def __dlpack__(self, stream: Any = None):
        """
        Converts Tensor to a DLManagedTensor.
        """
        # since the underlying data is numpy/cupy
        # we are reusing their __dlpack__() method
        array = self.eval().view()
        return array.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        array = self.eval().view()
        return array.__dlpack_device__()
