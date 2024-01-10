from typing import List

from tripy.backend.mlir.mlir import mlir_wrapper, void_ptr, ExecInitializerResult
from tripy.common.logging import G_LOGGER
from tripy.frontend import Tensor
from tripy.common.datatype import convert_tripy_to_numpy_dtype
from tripy.frontend.ops import Storage
from tripy.util import log_time


class FlatIRExecutor:
    """
    Manages the backend's executable.
    It can execute the executable generated by compiler with variable input tensors.
    The compiled executable is held by the class instance and freed when the instance is destructed.
    """

    def __init__(
        self, executable: void_ptr, output_devices: List, i_tensor_info: List = None, o_tensor_info: List = None
    ) -> None:
        """
        Args:
            executable: mlir executable generated by the compiler
            output_devices: output devices inferred by Trace
            i_tensor_info: Shape and data type info for inputs used to deduce runtime values
            o_tensor_info: Shape and data type info outputs used for output allocation
        """
        self.compiler = mlir_wrapper()
        assert executable is not None, "executable must be compiled!"
        self.executable = executable
        self.output_devices = output_devices
        self.i_tensor_info = i_tensor_info
        self.o_tensor_info = o_tensor_info

    def destroy(self):
        self.compiler.exec_destroy(self.executable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            info = (exc_type, exc_value, traceback)
            G_LOGGER.exception("Exception occurred in FlatIRExecutor", exc_info=info)

        # Destroy the allocations and the loadable executor.
        self.destroy()

        return False

    @log_time
    def execute(self, inputs: List[Tensor] = []) -> List[Storage]:
        """
        Executes the compiled MLIR program and returns the output of the computation as a list of numpy arrays.

        Args:
            inputs: a list of tripy Tensor for input tensors

        Returns:
            A list of Storage instances with data on self.output_devices
        """
        # Create execargs
        device_inputs = []
        for inp in inputs:
            if inp.const_fold:
                continue
            inp_storage = inp.op
            assert isinstance(inp_storage, Storage), "Input tensors must be evaluated!"
            if inp_storage.device.kind != "gpu":
                raise Exception("Input tensors must be on device!")
            device_inputs.append(inp_storage)
        exec_args = self.compiler.exec_initializer(
            self.executable, device_inputs, self.output_devices, self.i_tensor_info, self.o_tensor_info
        )

        # Execute and populate device pointers.
        self.compiler.execute(self.executable, exec_args)
        # Create a list to store the output arrays
        outputs: List[Storage] = []

        num_outputs: int = len(self.o_tensor_info)
        num_devices: int = 1  # Assuming 1 device, adjust as needed

        for i in range(num_devices):
            for j in range(num_outputs):
                index = i * num_outputs + j
                s = exec_args.outputs[index]
                # Let's return Storage and let user interpret it.
                outputs.append(s)

        return outputs
