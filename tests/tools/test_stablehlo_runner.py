import os
import tempfile

import numpy as np
import pytest

import tripy as tp
from tripy.frontend.trace import Trace
from tripy.backend.mlir.compiler import Compiler
from tools.stablehlo_runner import compile_code, read_program_from_file, preprocess_program


@pytest.fixture
def init_mlir_textual():

    # Ensure big matrix so that mlir_textual has constants hidden.
    a_np = np.random.rand(4).astype(np.float32)
    b_np = np.random.rand(2, 4).astype(np.float32)

    a = tp.Tensor(a_np)
    b = tp.Tensor(b_np)

    out = a + b
    trace = Trace([out])
    mlir_textual = Compiler.remove_stablehlo_constants(str(trace.to_flat_ir().to_mlir()))
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
