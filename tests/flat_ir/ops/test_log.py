import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import LogOp


@pytest.fixture
def flat_ir():
    tensor = tp.Tensor([1.0, 2.0], name="tensor")
    out = tensor.log()
    out.name = "out"

    trace = Trace([out])
    yield trace.to_flat_ir()


class TestLogOp:
    def test_str(self, flat_ir):
        Log = flat_ir.ops[-1]
        assert isinstance(Log, LogOp)
        assert str(Log) == "out: [shape=(2,), dtype=(float32), loc=(gpu:0)] = LogOp(tensor)"

    def test_mlir(self, flat_ir):
        helper.check_mlir(
            flat_ir.to_mlir(),
            """
            module {
                func.func @main() -> tensor<2xf32> {
                    %0 = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
                    %1 = stablehlo.log %0 : tensor<2xf32>
                    return %1 : tensor<2xf32>
                }
            }
            """,
        )
