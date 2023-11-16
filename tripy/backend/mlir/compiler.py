from typing import Tuple

from tripy.backend.mlir.mlir import mlir_wrapper, void_ptr, ExecInitializerResult
from tripy.flat_ir import FlatIR
from tripy.common.logging import G_LOGGER
from tripy.util import log_time
from tripy.util.util import prefix_with_line_numbers
from tripy.backend.mlir.mlir_translator import lower_flat_ir_to_mlir
from tripy.backend.mlir.types import TensorShape

import ctypes

from tripy.ops.storage import Storage


class FlatIRCompiler:
    """
    Represents the compiler for FlatIR, which converts FlatIR into a StableHLO representation
    and compiles it into an executable using mlir-tensorrt compiler.
    """

    def __init__(self, flat_ir: FlatIR) -> None:
        self.compiler = mlir_wrapper()
        self.flat_ir = flat_ir
        # Store allocations done at the init of executor
        self.exec_args: ExecInitializerResult = None
        self.executable: void_ptr = None

    def __enter__(self) -> Tuple[void_ptr, ExecInitializerResult]:
        self.executable = self.compile(self.flat_ir)

        # Inputs are already allocated on device. Copy input shapes and device pointers.
        inputs = []
        for i in range(len(self.flat_ir.inputs)):
            inp_storage = self.flat_ir.inputs[i][1]
            assert inp_storage.device.kind == "gpu", "Input tensors must be on device!"
            s = inp_storage.to_ctypes()
            inputs.append(s)

        # Populate output tensor shapes and device memory pointers.
        self.exec_args = self.compiler.exec_initializer(self.executable, inputs)

        return self.executable, self.exec_args

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            info = (exc_type, exc_value, traceback)
            G_LOGGER.exception("Exception occurred in FlatIRCompiler", exc_info=info)

        # Destroy the allocations and the loadable executor.
        self.compiler.exec_destroy(self.executable)

        return False

    @log_time
    def compile(self, flat_ir: FlatIR) -> void_ptr:
        """
        Given a FlatIR, compile function traces the computation graph and generates a mlir-tensorrt executable.

        Args:
            flat_ir: FlatIR representation of the program that needs to be compiled.
        Returns:
            Pointer to the executable generated by the mlir-tensorrt compiler.
        """
        # Lower flatIR to corresponding StableHLO IR.
        mlir_module = lower_flat_ir_to_mlir(flat_ir)
        mlir_textual = mlir_module.__str__()
        G_LOGGER.ir_printer(f"StableHLO IR:\n{prefix_with_line_numbers(mlir_textual)}")
        return self.compiler.compile(mlir_textual)
