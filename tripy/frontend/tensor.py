from typing import List

from tripy import utils
from tripy.common.array import Array
from tripy.common.logging import G_LOGGER
from tripy.frontend.ops import TENSOR_METHOD_REGISTRY, Storage


class TensorMeta(type):
    def __new__(cls, name, bases, dct):
        new = type.__new__(cls, name, bases, dct)

        # Add methods specified by individual ops to this class.
        for name in TENSOR_METHOD_REGISTRY:
            setattr(new, name, TENSOR_METHOD_REGISTRY[name])

        return new


class Tensor(metaclass=TensorMeta):
    """
    Represents a lazily evaluated tensor.
    """

    _COUNT = 0

    def _finalize(self, inputs: List["Tensor"], OpType: type, *args, **kwargs) -> None:
        # It is very important that this is called from all entrypoints to creating a tensor.
        # We include logic here that needs to be applied to all tensors.
        from tripy.frontend.trace.tensor import TraceTensor

        def get_name():
            name = f"t{Tensor._COUNT}"
            Tensor._COUNT += 1
            return name

        self._stack_info = utils.get_stack_info()

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
        tensor = Tensor()
        tensor._finalize(inputs, OpType, *args, **kwargs)
        return tensor

    def eval(self) -> Array:
        from tripy.backend.mlir.compiler import FlatIRCompiler
        from tripy.backend.mlir.executor import FlatIRExecutor
        from tripy.frontend.trace import Trace

        if isinstance(self.op, Storage):
            return self.op.data

        trace = Trace([self])
        G_LOGGER.ir_printer(f"Trace :\n{trace}")
        flat_ir = trace.to_flat_ir()
        G_LOGGER.ir_printer(f"FlatIR :\n{flat_ir}")
        output_devices = [o.device for o in trace.outputs]
        i_tensor_info, o_tensor_info = flat_ir.io_tensor_info()
        compiler = FlatIRCompiler()
        executable = compiler.compile(flat_ir)
        with FlatIRExecutor(executable, output_devices, i_tensor_info, o_tensor_info) as executor:
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
        return f"tensor({self.eval().view()}, dtype={self.op.dtype}, loc={self.op.device})"
