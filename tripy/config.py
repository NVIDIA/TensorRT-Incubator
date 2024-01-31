import os
import tempfile

# backend/mlir configs
MLIR_LIB_PATH = os.path.join("/", "usr", "lib", "mlir-tensorrt")
MLIR_LIB_NAME = "libtripy_backend*.so"


# IR printing
CONSTANT_IR_PRINT_VOLUME_THRESHOLD = 5
"""The volume threshold above which constants should not be printed in the IR"""
