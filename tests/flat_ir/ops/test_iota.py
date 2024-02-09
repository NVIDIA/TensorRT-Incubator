import pytest

import tripy as tp
from tests import helper
from tripy.flat_ir.ops import IotaOp
from tripy.frontend.trace import Trace


@pytest.fixture
def flat_ir():
    out = tp.iota((2, 3))
    out.name = "out"

    trace = Trace([out])
    yield trace.to_flat_ir()


class TestIotaOp:
    def test_str(self, flat_ir):
        iota = flat_ir.ops[-1]
        assert isinstance(iota, IotaOp)
        assert (
            str(iota)
            == "out: [shape=(2, 3,), dtype=(float32), loc=(gpu:0)] = IotaOp(dim=0, shape=(2, 3), dtype=float32)"
        )

    def test_mlir(self, flat_ir):
        helper.check_mlir(
            flat_ir.to_mlir(),
            """
            module {
                func.func @main() -> tensor<2x3xf32> {
                    %0 = stablehlo.iota dim = 0 : tensor<2x3xf32>
                    return %0 : tensor<2x3xf32>
                }
            }
            """,
        )
