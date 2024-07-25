import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ConstantOp


class TestConstantOp:
    def test_str(self):
        out = tp.Tensor([2.0, 3.0], dtype=tp.float32, name="out")

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        const = flat_ir.ops[-1]
        assert isinstance(const, ConstantOp)
        assert (
            str(const)
            == "out: [rank=(1), shape=((2,)), dtype=(float32), loc=(gpu:0)] = ConstantOp(data=[2.0000, 3.0000])"
        )

    def test_mlir(self):
        out = tp.Tensor([2, 3], dtype=tp.int32, name="out")

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()
        mlir_text = str(flat_ir.to_mlir())
        target = "%c = stablehlo.constant dense<[2, 3]> : tensor<2xi32>"
        assert target in mlir_text

    def test_mlir_bool(self):
        # we need to create a bool constant with an int constant and then cast because MLIR does not allow
        # for bools in dense array attrs
        out = tp.Tensor([True, False], dtype=tp.bool, name="out")

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()
        mlir_text = str(flat_ir.to_mlir())

        int_constant = "%c = stablehlo.constant dense<[1, 0]> : tensor<2xi32>"
        conversion = "%0 = stablehlo.convert %c : (tensor<2xi32>) -> tensor<2xi1>"
        assert int_constant in mlir_text
        assert conversion in mlir_text
