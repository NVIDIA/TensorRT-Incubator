from typing import List

from tripy import util
from tripy.common.logging import G_LOGGER
from tripy.ops import TENSOR_METHOD_REGISTRY


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

    def _finalize(self, inputs, op) -> None:
        # It is very important that this is called from all entrypoints to creating a tensor.
        # We include logic here that needs to be applied to all tensors.
        self.inputs = inputs
        self.op = op
        self._stack_info = util.get_stack_info()
        # Const fold the tensor in JIT functions
        # Tensor will be treated as an input if set to False
        self.const_fold = True

    @staticmethod
    def build(inputs: "List[Tensor]", op: "tripy.ops.BaseOperator") -> None:
        tensor = Tensor()
        tensor._finalize(inputs, op)
        return tensor

    def eval(self) -> None:
        from tripy.backend.mlir.compiler import FlatIRCompiler
        from tripy.backend.mlir.executor import FlatIRExecutor
        from tripy.flat_ir import FlatIR
        from tripy.ops import Storage
        from tripy.common import device

        if isinstance(self.op, Storage):
            return self.op.data

        flat_ir = FlatIR([self])
        G_LOGGER.ir_printer(f"flatIR :\n{flat_ir}")
        output_devices = [o.device for o in flat_ir.outputs]

        compiler = FlatIRCompiler()
        with FlatIRExecutor(compiler.compile(flat_ir), output_devices) as executor:
            # Upon computing the value of this tensor, we switch it to have a `Storage`
            # parameter so that it does not need to be computed again.
            value = executor.execute()
            self.inputs = []
            assert len(value) == 1, "Expects only one output from MLIR executor"
            self.op = Storage(value[0], device=output_devices[0])
            return value[0]

    def __repr__(self) -> str:
        return f"tensor({self.eval()}, dtype={self.op.data.dtype}, loc={self.op.device})"
