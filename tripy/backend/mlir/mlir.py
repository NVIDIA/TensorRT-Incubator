import atexit
import ctypes
from jinja2 import Template

from tripy.logging import G_LOGGER
from tripy.util.util import find_file_in_dir
from tripy.util import log_time

void_ptr = ctypes.c_void_p
char_ptr = ctypes.c_char_p
c_int = ctypes.c_int
POINTER = ctypes.POINTER


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
        lib_path = find_file_in_dir("libtripy_backend*.so", "/usr/lib/mlir-tensorrt/")
        assert (
            len(lib_path) == 1
        ), f"Compiler expects exactly 1 tripy backend library to be available.  Found {len(lib_path)} libraries."
        self.compiler_lib = ctypes.CDLL(lib_path[0])

        # Entry points of the MLIR compiler.
        self.mlir_initialize = func_wrapper(self.compiler_lib, "initialize", [], void_ptr)
        self.mlir_destroy = func_wrapper(self.compiler_lib, "destroy", [void_ptr], None)
        self.mlir_compile = func_wrapper(self.compiler_lib, "compile", [void_ptr, char_ptr, c_int], void_ptr)
        self.mlir_execute = func_wrapper(self.compiler_lib, "execute", [void_ptr, POINTER(void_ptr), void_ptr], None)
        self.mlir_executor_initialize = func_wrapper(
            self.compiler_lib, "loadedExecInitializer", [void_ptr, void_ptr], void_ptr
        )
        self.mlir_executor_destroy = func_wrapper(self.compiler_lib, "loadedExecDestructor", [void_ptr, void_ptr], None)

        self.compiler = self.mlir_initialize()
        if not self.compiler:
            G_LOGGER.critical("Could not load the backend compiler.")

    def destroy(self):
        """
        Calls the MLIR compiler destructor to free up allocated memory.
        """
        self.mlir_destroy(void_ptr(self.compiler))

    def exec_initializer(self, executable: void_ptr, execargs: void_ptr):
        """
        Calls the initializer for loadable executable.
        """
        return self.mlir_executor_initialize(executable, execargs)

    def exec_destroy(self, executable: void_ptr, execargs: void_ptr):
        """
        Calls the destructor for loadable executable.
        """
        self.mlir_executor_destroy(executable, execargs)

    def compile(self, code: str) -> void_ptr:
        """
        Args:
            code: MLIR program that needs to be compiled.
        Returns:
            Pointer to the executable
        """
        return self.mlir_compile(void_ptr(self.compiler), code.encode(), len(code))

    def execute(self, executable: void_ptr, dst, exec_args):
        """
        Args:
            executable: MLIR executable.
            exec_args: execution arguments.
        """

        self.mlir_execute(void_ptr(executable), dst, exec_args)


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
