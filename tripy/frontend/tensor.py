from typing import List, Any

from tripy import util
from tripy.logging import G_LOGGER
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

    def __init__(self, data: Any) -> None:
        if data is not None:
            from tripy.ops import Storage

            self._finalize([], Storage(data))

    @staticmethod
    def build(inputs: "List[Tensor]", op: "tripy.ops.BaseOperator") -> None:
        """
        Args:
            inputs: The inputs to this tensor.
            op: The operation being applied.
        """
        tensor = Tensor(data=None)
        tensor._finalize(inputs, op)
        return tensor

    def eval(self) -> None:
        from tripy.flat_ir import FlatIR
        from tripy.backend.mlir.compiler import FlatIRCompiler
        from tripy.backend.mlir.executor import FlatIRExecutor
        from tripy.ops import Storage

        if isinstance(self.op, Storage):
            return self.op.data

        flat_ir = FlatIR([self])
        G_LOGGER.ir_printer(f"flatIR :\n{flat_ir}")

        with FlatIRCompiler() as compiler, FlatIRExecutor(flat_ir) as executor:
            executable = compiler.compile(flat_ir)
            # Upon computing the value of this tensor, we switch it to have a `Storage`
            # parameter so that it does not need to be computed again.
            value = executor.execute(executable)
            self.inputs = []
            self.op = Storage(value)
            return value

    def __repr__(self) -> str:
        return f"{self.eval()}"
