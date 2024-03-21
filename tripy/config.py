import tempfile
import os
from tripy.utils import export

# IR printing
CONSTANT_IR_PRINT_VOLUME_THRESHOLD = 5
"""The volume threshold above which constants should not be printed in the IR"""

# MLIR Debug options
MLIR_DEBUG_ENABLED = os.environ.get("TRIPY_MLIR_DEBUG_ENABLED", "0") == "1"
MLIR_DEBUG_TYPES = ["-mlir-print-ir-after-all"]
MLIR_DEBUG_TREE_PATH = os.path.join("/", "tripy", "mlir-dumps")

# Compiler Options
temp_dir_path = tempfile.gettempdir()

# Variables that are exposed to the user are kept lowercase.
timing_cache_file_path = os.path.join(temp_dir_path, "tripy-cache")
