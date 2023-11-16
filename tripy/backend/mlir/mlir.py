import atexit
from collections import namedtuple
import ctypes

from tripy.common.logging import G_LOGGER
from tripy.util import log_time
from tripy.util.util import find_file_in_dir

from tripy.ops.allocator import GpuAllocator
from tripy.common.datatype import *  # Assuming you have specific types defined in this module
from tripy.backend.mlir.types import *  # Assuming you have specific types defined in this module

# Define a namedtuple to hold the result of the execution initializer
ExecInitializerResult = namedtuple("ExecInitializerResult", ["input_shapes", "inputs", "output_shapes", "outputs"])


def func_wrapper(lib, c_func, argtypes, restype):
    """
    Set the argument and return types using ctypes and return the wrapped function.
    """
    func = lib.__getattr__(c_func)
    func.argtypes = argtypes
    func.restype = restype
    return func


class _MlirCompiler:
    """
    Represents the interface with MLIR compiler.
    Do not call this class directly as it would lead to multiple MLIR initialization which can cause program to abort.
    Instead of a singleton class implementation, tripy assumes that `_MlirCompiler` will be imported from other modules.
    """

    @log_time
    def __init__(self) -> None:
        # TODO: Make this not use a hard-coded path.
        lib_path = find_file_in_dir("libtripy_backend*.so", "/tripy/mlir-tensorrt/build/lib/Integrations/tripy")
        assert (
            len(lib_path) == 1
        ), f"Compiler expects exactly 1 tripy backend library to be available.  Found {len(lib_path)} libraries."
        self.compiler_lib = ctypes.CDLL(lib_path[0])

        # Entry points of the MLIR compiler.
        self.mlir_initialize = func_wrapper(self.compiler_lib, "initialize", [], void_ptr)
        self.mlir_destroy = func_wrapper(self.compiler_lib, "destroy", [void_ptr], None)
        self.mlir_compile = func_wrapper(self.compiler_lib, "compile", [void_ptr, char_ptr, c_int], void_ptr)
        self.mlir_execute = func_wrapper(
            self.compiler_lib,
            "execute",
            [
                void_ptr,
                c_int,
                c_int,
                ctypes.POINTER(TensorShape),
                void_ptr,
                c_int,
                ctypes.POINTER(TensorShape),
                void_ptr,
            ],
            None,
        )
        self.mlir_load_exec_init = func_wrapper(
            self.compiler_lib,
            "loadedExecInitializer",
            [
                void_ptr,
                ctypes.POINTER(TensorShape),
                ctypes.POINTER(ctypes.c_int),
            ],
            None,
        )
        self.mlir_executor_destroy = func_wrapper(self.compiler_lib, "loadedExecDestructor", [void_ptr], None)

        self.compiler = self.mlir_initialize()
        if not self.compiler:
            G_LOGGER.critical("Could not load the backend compiler.")

        self.allocator = GpuAllocator()

    def destroy(self):
        """
        Calls the MLIR compiler destructor to free up allocated memory.
        """
        self.mlir_destroy(void_ptr(self.compiler))
        self.allocator = None

    def exec_initializer(self, executable: void_ptr, inputs) -> ExecInitializerResult:
        """
        Calls the initializer for a loadable executable.

        Args:
            executable (void_ptr): Pointer to the MLIR executable.
            allocator (void_ptr): Allocator for memory allocation.

        Returns:
            ExecInitializerResult: A named tuple containing result buffers, shapes, num_devices, and num_outputs.
        """
        # Call the function and receive the pointers to the arrays and counts
        nb_outputs = ctypes.c_int()

        self.mlir_load_exec_init(executable, None, ctypes.byref(nb_outputs))
        output_shapes_arr = (TensorShape * nb_outputs.value)()
        input_shapes_arr = (TensorShape * len(inputs))()

        self.mlir_load_exec_init(executable, output_shapes_arr, ctypes.byref(nb_outputs))

        # Allocate output memory and store buffer pointers.
        outputs = [self.allocator.allocate_async(shape) for shape in output_shapes_arr]

        return ExecInitializerResult(input_shapes_arr, inputs, output_shapes_arr, outputs)

    def exec_destroy(self, executable: void_ptr):
        """
        Calls the destructor for loadable executable.
        """
        self.mlir_executor_destroy(executable)

    def compile(self, code: str) -> void_ptr:
        """
        Args:
            code: MLIR program that needs to be compiled.
        Returns:
            Pointer to the executable
        """
        return self.mlir_compile(void_ptr(self.compiler), code.encode(), len(code))

    def execute(self, executable: void_ptr, exec_args: ExecInitializerResult) -> None:
        """
        Execute the MLIR executable with the provided execution arguments.

        Args:
            executable (void_ptr): A pointer to the MLIR executable that will be executed.
            exec_args (ExecInitializerResult): A named tuple containing the result of the execution
                initializer, including buffers, sizes, the number of devices, and the number of outputs.
                - exec_args.inputs: A list of Storage objects representing input buffers.
                - exec_args.outputs: A list of Storage objects representing output buffers.
                - exec_args.input_shapes: An array of input shapes.
                - exec_args.output_shapes: An array of output shapes.
        """
        # Create ctypes compatible device memory void pointers from result buffers.

        in_mem_ptrs = (
            (ctypes.c_void_p * len(exec_args.inputs))(*(r.data.ptr for r in exec_args.inputs))
            if len(exec_args.inputs) > 0
            else None
        )
        out_mem_ptrs = (
            (ctypes.c_void_p * len(exec_args.outputs))(*(r.data.mem.ptr for r in exec_args.outputs))
            if len(exec_args.outputs) > 0
            else None
        )

        in_len = exec_args.input_shapes._length_ if len(exec_args.input_shapes) > 0 else 0
        out_len = exec_args.output_shapes._length_ if len(exec_args.output_shapes) > 0 else 0

        self.mlir_execute(
            void_ptr(executable),
            1,  # Assuming 1 device
            in_len,
            exec_args.input_shapes,
            in_mem_ptrs,
            out_len,
            exec_args.output_shapes,
            out_mem_ptrs,
        )


G_COMPILER_BACKEND = None


def mlir_wrapper():
    """
    Returns the global MLIR Compiler.

    Returns:
        Compiler: The global MLIR compiler.
    """
    global G_COMPILER_BACKEND
    if G_COMPILER_BACKEND is None:
        G_COMPILER_BACKEND = _MlirCompiler()
    return G_COMPILER_BACKEND


def mlir_close():
    if G_COMPILER_BACKEND is not None:
        G_COMPILER_BACKEND.destroy()


atexit.register(mlir_close)
