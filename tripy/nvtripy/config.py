#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Global configuration options for Tripy.
"""

import os
import sys
import tempfile

from nvtripy import export

export.public_api(autodoc_options=[":no-members:", ":no-special-members:"])(sys.modules[__name__])

# MLIR-TRT Debug options
enable_mlir_debug = os.environ.get("TRIPY_MLIR_DEBUG_ENABLED", "0") == "1"
mlir_debug_types = os.environ.get("TRIPY_MLIR_DEBUG_TYPES", "-mlir-print-ir-after-all,-translate-to-tensorrt").split(
    ","
)
mlir_debug_tree_path = os.environ.get("TRIPY_MLIR_DEBUG_PATH", os.path.join("/", "nvtripy", "mlir-dumps"))

# TensorRT debug options
enable_tensorrt_debug = os.environ.get("TRIPY_TRT_DEBUG_ENABLED", "0") == "1"
tensorrt_debug_path = os.environ.get("TRIPY_TRT_DEBUG_PATH", os.path.join("/", "nvtripy", "tensorrt-dumps"))

timing_cache_file_path: str = export.public_api(
    document_under="config.rst",
    autodoc_options=[":no-value:"],
    module=sys.modules[__name__],
    symbol="timing_cache_file_path",
)(os.path.join(tempfile.gettempdir(), "nvtripy-cache"))
"""Path to a timing cache file that can be used to speed up compilation time."""

enable_dtype_checking: bool = export.public_api(
    document_under="config.rst",
    module=sys.modules[__name__],
    symbol="enable_dtype_checking",
)(True)
"""Whether to enable data type checking in API functions."""
