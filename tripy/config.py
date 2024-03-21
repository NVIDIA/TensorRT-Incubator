"""
Controls global configuration options for Tripy.
"""

import os
import sys
import tempfile

from tripy import export

export.public_api(autodoc_options=[":members:"])(sys.modules[__name__])

# MLIR Debug options
MLIR_DEBUG_ENABLED = os.environ.get("TRIPY_MLIR_DEBUG_ENABLED", "0") == "1"
MLIR_DEBUG_TYPES = ["-mlir-print-ir-after-all"]
MLIR_DEBUG_TREE_PATH = os.path.join("/", "tripy", "mlir-dumps")

# Compiler Options
temp_dir_path = tempfile.gettempdir()

# Variables that are exposed to the user are kept lowercase.
timing_cache_file_path = os.path.join(temp_dir_path, "tripy-cache")
