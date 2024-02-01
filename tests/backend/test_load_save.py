import os
import tempfile

import numpy as np
import pytest

import tripy as tp
from tripy.backend.jit.utils import get_tensor_info
from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor
from tripy.frontend.trace import Trace


@pytest.fixture
def init_flat_ir():
    a = tp.Tensor([1, 2])
    b = tp.Tensor([2, 3])
    out = a + b
    trace = Trace([out])
    flat_ir = trace.to_flat_ir()
    return flat_ir


def test_save_load_from_file(init_flat_ir):
    with tempfile.NamedTemporaryFile() as temp_file:
        filename = temp_file.name
        flat_ir = init_flat_ir

        compiler = FlatIRCompiler()
        executable = compiler.compile(flat_ir)
        compiler.compiler.save(executable, filename)
        assert os.path.exists(filename)

        executable = compiler.compiler.load(filename)
        with FlatIRExecutor(executable, get_tensor_info(flat_ir.inputs), get_tensor_info(flat_ir.outputs)) as executor:
            out = executor.execute()
            assert len(out) == 1
            assert (out[0].view().get() == np.array([3, 5])).all()


def test_save_load_from_string(init_flat_ir):
    with tempfile.NamedTemporaryFile() as temp_file:
        filename = temp_file.name
        flat_ir = init_flat_ir
        compiler = FlatIRCompiler()
        executable = compiler.compile(flat_ir)
        compiler.compiler.save(executable, filename)
        assert os.path.exists(filename)

        exec_str = temp_file.read()
        executable = compiler.compiler.load(data=exec_str)
        with FlatIRExecutor(executable, get_tensor_info(flat_ir.inputs), get_tensor_info(flat_ir.outputs)) as executor:
            out = executor.execute()
            assert len(out) == 1
            assert (out[0].view().get() == np.array([3, 5])).all()
