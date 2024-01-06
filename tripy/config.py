import os
import tempfile

# backend/mlir configs
MLIR_LIB_PATH = os.path.join("/", "usr", "lib", "mlir-tensorrt")
MLIR_LIB_NAME = "libtripy_backend*.so"


# JIT configs
# Fixed length of jit function's hash
JIT_CACHE_HASH_LENGTH = 30
# Parent directory to save cached engines
JIT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "tripy", "jit_caches")
