import atexit
import ctypes
from jinja2 import Template

from tripy.logging import G_LOGGER
from tripy.util.util import find_file_in_dir
from tripy.util import log_time

void_ptr = ctypes.c_void_p
char_ptr = ctypes.c_char_p
c_int = ctypes.c_int


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
        lib_path = find_file_in_dir("libtripy_backend*.so", "/tripy/mlir-tensorrt/build/")
        assert len(lib_path) == 1, "Compiler expects only 1 tripy backend library to be available."
        self.compiler_lib = ctypes.CDLL(lib_path[0])

        # Entry points of the MLIR compiler.
        self.mlir_initialize = func_wrapper(self.compiler_lib, "initialize", [], void_ptr)
        self.mlir_destroy = func_wrapper(self.compiler_lib, "destroy", [void_ptr], None)
        self.mlir_compile = func_wrapper(self.compiler_lib, "compile", [void_ptr, char_ptr, c_int], void_ptr)
        self.mlir_execute = func_wrapper(self.compiler_lib, "execute", [void_ptr, void_ptr], None)

        self.compiler = self.mlir_initialize()
        if not self.compiler:
            G_LOGGER.critical("Could not load the backend compiler.")

    def destroy(self):
        """
        Calls the MLIR compiler destructor to free up allocated memory.
        """
        self.mlir_destroy(void_ptr(self.compiler))

    def compile(self, code: str) -> void_ptr:
        """
        Args:
            code: MLIR program that needs to be compiled.
        Returns:
            Pointer to the executable
        """
        return self.mlir_compile(void_ptr(self.compiler), code.encode(), len(code))

    def execute(self, executable: void_ptr, output_ptr):
        """
        Args:
            executable: MLIR executable.
            output_ptr: Address of the output array.
        """
        self.mlir_execute(void_ptr(executable), output_ptr)


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
