import pytest
import numpy as np

import tripy
from tripy import jit
from tripy.frontend import Dim, Tensor
from tripy.frontend.trace import Trace
from tripy.backend.mlir.mlir_translator import lower_flat_ir_to_mlir
from tripy.utils.utils import prefix_with_line_numbers


@pytest.mark.skip(reason="Dynamic shape is not working with MLIR backend yet.")
def test_mlir_dynamic_generated():
    dim = Dim(2, min=2, opt=2, max=4)
    a = Tensor(np.ones((2, 3), dtype=np.float32), shape=(dim, 3), device=tripy.device("gpu"))
    b = Tensor(np.ones((2, 3), dtype=np.float32), shape=(dim, 3), device=tripy.device("gpu"))

    @jit
    def func(a, b):
        c = a + b
        return c

    out = func(a, b)
    trace = Trace(out)
    flat_ir = trace.to_flat_ir()
    mlir_module = lower_flat_ir_to_mlir(flat_ir)
    mlir_textual = prefix_with_line_numbers(mlir_module.__str__())
    assert "%0 = stablehlo.add %arg0, %arg1 : tensor<?x3xf32>" in mlir_textual
