import numpy as np

from tripy.frontend import Tensor
from tripy.flat_ir import FlatIR
from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor


class TestFunctional:
    def test_add_two_tensors(self):
        arr = np.array([2, 3], dtype=np.float32)
        a = Tensor(arr)
        b = Tensor(np.ones(2, dtype=np.float32))

        c = a + b
        out = c + c
        assert (out.eval() == np.array([6.0, 8.0])).all()

    def test_multi_output_flat_ir(self):
        shape = 2
        a = Tensor(np.ones(shape))
        b = Tensor(np.ones(shape))
        c = a + b
        d = c + c
        flat_ir = FlatIR([c, d])

        with FlatIRCompiler() as compiler, FlatIRExecutor(flat_ir) as executor:
            executable = compiler.compile(flat_ir)
            out = executor.execute(executable)
            assert len(out) == 2 and (out[0] == np.array([2.0, 2.0])).all() and (out[1] == np.array([4.0, 4.0])).all()
