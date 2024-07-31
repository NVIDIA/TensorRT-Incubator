#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import tempfile

import cupy as cp
import pytest

import tripy as tp
from tools.stablehlo_runner import compile_code, preprocess_program, read_program_from_file
from tripy.backend.mlir.utils import remove_constants
from tripy.frontend.trace import Trace


@pytest.fixture
def init_mlir_textual():

    # Ensure big matrix so that mlir_textual has constants hidden.
    a_cp = cp.arange(4).astype(cp.float32)
    b_cp = cp.arange(8).reshape((2, 4)).astype(cp.float32)

    a = tp.Tensor(a_cp)
    b = tp.Tensor(b_cp)

    out = a + b
    trace = Trace([out])
    mlir_textual = remove_constants(str(trace.to_flat_ir().to_mlir()))
    return mlir_textual


def test_mlir_tool(init_mlir_textual):
    with tempfile.NamedTemporaryFile(mode="w+") as temp_file:
        filename = temp_file.name
        mlir_textual = init_mlir_textual
        # Write the program to the temporary file
        temp_file.write(mlir_textual)
        temp_file.flush()  # Ensure all data is written to disk
        assert os.path.exists(filename)

        cleaned_code = preprocess_program(read_program_from_file(filename))
        # If the program compilation fails, currently it causes python to crash (Fatal Python error: Aborted)
        compile_code(cleaned_code)
