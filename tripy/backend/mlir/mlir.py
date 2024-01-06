import atexit
import ctypes
import cupy as cp
import os

from collections import namedtuple

from tripy import config
from tripy.common.logging import G_LOGGER
from tripy.util import log_time
from tripy.util.util import find_file_in_dir

from tripy.common.ctypes import void_ptr, char_ptr, c_int, TensorShape, convert_mlirdtype_to_tripy_dtype
from tripy.frontend.ops import Storage

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
        lib_path = find_file_in_dir(config.MLIR_LIB_NAME, mlir_lib_path())
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
            ],
            None,
        )
        self.mlir_executor_destroy = func_wrapper(self.compiler_lib, "loadedExecDestructor", [void_ptr], None)
        self.mlir_save_executable = func_wrapper(self.compiler_lib, "saveExecutable", [void_ptr, char_ptr], None)
        self.mlir_load_executable_from_file = func_wrapper(
            self.compiler_lib, "loadExecutableFromFile", [char_ptr], void_ptr
        )
        self.mlir_load_executable_from_string = func_wrapper(
            self.compiler_lib, "loadExecutableFromString", [char_ptr], void_ptr
        )

        self.compiler = self.mlir_initialize()
        if not self.compiler:
            G_LOGGER.critical("Could not load the backend compiler.")

    def destroy(self):
        """
        Calls the MLIR compiler destructor to free up allocated memory.
        """
        self.mlir_destroy(void_ptr(self.compiler))
        self.allocator = None

    def exec_initializer(self, executable: void_ptr, inputs, output_devices) -> ExecInitializerResult:
        """
        Calls the initializer for a loadable executable.

        Args:
            executable (void_ptr): Pointer to the MLIR executable.
            inputs: A list of input Storage objects
            output_devices: A list of output devices

        Returns:
            ExecInitializerResult: A named tuple containing input buffer, output buffer, and output shapes.
        """
        nb_outputs = len(output_devices)
        output_shapes_arr = (TensorShape * nb_outputs)()

        self.mlir_load_exec_init(executable, output_shapes_arr)

        # Allocate output memory and store buffer pointers.
        outputs = [
            Storage(
                None,
                shape=shape.get_shape_arr(),
                dtype=convert_mlirdtype_to_tripy_dtype(shape.dtype),
                device=out_device,
            )
            for shape, out_device in zip(output_shapes_arr, output_devices)
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
                            else r.data.byte_buffer.ctypes.data
                        ),
                        data,
                    )
                )
                if len(data) > 0
                else None
            )

        output_device_arr = [1 if out.device.kind == "gpu" else 0 for out in exec_args.outputs]
        output_devices = (ctypes.c_int * len(exec_args.outputs))(*output_device_arr)

        self.mlir_execute(
            void_ptr(executable),
            get_mem_ptrs(exec_args.inputs),
            get_mem_ptrs(exec_args.outputs),
            output_devices,
        )

    def save(self, executable: void_ptr, filename: str) -> None:
        """
        Saves the MLIR executable to the given file

        Args:
            executable: Pointer to the MLIR executable
            filename: A string of the file name
        """
        return self.mlir_save_executable(void_ptr(executable), filename.encode())

    def load(self, path: str = None, data: bytes = None) -> void_ptr:
        """
        Loads the MLIR executable from the source

        Args:
            path: A string of the file name to load the executable
            data: Byte data to load the executable
        """
        if path is not None:
            if not os.path.exists(path):
                raise Exception("File not found")
            return self.mlir_load_executable_from_file(path.encode())
        elif data is not None:
            return self.mlir_load_executable_from_string(data)
        else:
            raise Exception("One of path/data must be given")


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

    path = custom_integ_path or config.MLIR_LIB_PATH
    return path


atexit.register(mlir_close)
