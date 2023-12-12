import atexit
import ctypes
import cupy as cp
import os

from collections import namedtuple

from tripy.common.logging import G_LOGGER
from tripy.util import log_time
from tripy.util.util import find_file_in_dir

from tripy.common.ctypes import void_ptr, char_ptr, c_int, TensorShape, convert_mlirdtype_to_tripy_dtype
from tripy.ops.storage import Storage

# Define a namedtuple to hold the result of the execution initializer
ExecInitializerResult = namedtuple("ExecInitializerResult", ["inputs", "output_shapes", "outputs"])


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
        lib_path = find_file_in_dir("libtripy_backend*.so", mlir_lib_path())
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
                void_ptr,
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
            ExecInitializerResult: A named tuple containing input buffer, output buffer, and output shapes.
        """
        # Call the function and receive the pointers to the arrays and counts
        nb_outputs = ctypes.c_int()

        self.mlir_load_exec_init(executable, None, ctypes.byref(nb_outputs))
        output_shapes_arr = (TensorShape * nb_outputs.value)()

        self.mlir_load_exec_init(executable, output_shapes_arr, ctypes.byref(nb_outputs))

        # Allocate output memory and store buffer pointers.
        from tripy.common.device import device as make_device

        outputs = [
            Storage(None, shape.get_shape_arr(), convert_mlirdtype_to_tripy_dtype(shape.dtype), make_device("gpu:0"))
            for shape in output_shapes_arr
        ]

        return ExecInitializerResult(inputs, output_shapes_arr, outputs)

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
                - exec_args.output_shapes: An array of output shapes.
        """

        # Create ctypes compatible device memory void pointers for i/o buffers.
        def get_mem_ptrs(data):
            return (
                (ctypes.c_void_p * len(data))(
                    *map(
                        lambda r: (
                            r.data.byte_buffer.data.ptr
                            if isinstance(r.data.byte_buffer.data, cp.cuda.memory.MemoryPointer)
                            else r.data.byte_buffer.data
                        ),
                        data,
                    )
                )
                if len(data) > 0
                else None
            )

        self.mlir_execute(
            void_ptr(executable),
            get_mem_ptrs(exec_args.inputs),
            get_mem_ptrs(exec_args.outputs),
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


def mlir_lib_path():
    custom_integ_path = os.getenv("MLIR_TRT_INTEGRATION_PATH")
    if custom_integ_path:
        G_LOGGER.info(f"Trying to build with custom mlir backend at {custom_integ_path}")

    path = custom_integ_path or "/usr/lib/mlir-tensorrt/"
    return path


atexit.register(mlir_close)
