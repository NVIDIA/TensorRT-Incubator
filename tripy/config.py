"""
Global configuration options for Tripy.
"""

import os
import sys
import tempfile

from tripy import export

export.public_api(autodoc_options=[":no-members:", ":no-special-members:"])(sys.modules[__name__])

# MLIR Debug options
enable_mlir_debug = os.environ.get("TRIPY_MLIR_DEBUG_ENABLED", "0") == "1"
mlir_debug_types = ["-mlir-print-ir-after-all"]
mlir_debug_tree_path = os.environ.get("TRIPY_MLIR_DEBUG_PATH", os.path.join("/", "tripy", "mlir-dumps"))

# Tensorrt debug options
enable_tensorrt_debug = os.environ.get("TRIPY_TRT_DEBUG_ENABLED", "0") == "1"
tensorrt_debug_path = os.environ.get("TRIPY_TRT_DEBUG_PATH", os.path.join("/", "tripy", "tensorrt-dumps"))

# Variables that are exposed to the user are kept lowercase.
timing_cache_file_path: str = export.public_api(
    document_under="config.rst",
    autodoc_options=[":no-value:"],
    module=sys.modules[__name__],
    symbol="timing_cache_file_path",
)(os.path.join(tempfile.gettempdir(), "tripy-cache"))
"""Path to a timing cache file that can be used to speed up compilation time"""
