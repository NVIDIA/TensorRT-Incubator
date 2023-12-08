import numpy as np
import pytest

from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor
from tripy.common.device import device
from tripy.flat_ir import FlatIR
from tripy.frontend import Tensor


class TestFunctional:
    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_add_two_tensors(self, kind):
        arr = np.array([2, 3], dtype=np.float32)
        a = Tensor(arr, device=device(kind))
        b = Tensor(np.ones(2, dtype=np.float32), device=device(kind))

        c = a + b
        out = c + c
        # TODO(#36): Implement comparison by mlir-tensorrt
        out = out.eval()
        if kind == "gpu":
            out = out.get()
        assert (out == np.array([6.0, 8.0])).all()

    def test_multi_output_flat_ir(self):
        shape = 2
        a = Tensor(np.ones(shape))
        b = Tensor(np.ones(shape))
        c = a + b
        d = c + c
        flat_ir = FlatIR([c, d])
        output_devices = [o.device for o in flat_ir.outputs]

        compiler = FlatIRCompiler()
        with FlatIRExecutor(compiler.compile(flat_ir), output_devices) as executor:
            out = executor.execute()
            assert len(out) == 2 and (out[0] == np.array([2.0, 2.0])).all() and (out[1] == np.array([4.0, 4.0])).all()
