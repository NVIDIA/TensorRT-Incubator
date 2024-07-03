import pytest

import tripy as tp
from tests import helper
from tripy.flat_ir.ops import DynamicIotaOp
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
        assert isinstance(iota, DynamicIotaOp)
        assert (
            str(iota) == "out: [rank=(2), shape=(?, ?,), dtype=(float32), loc=(gpu:0)] = DynamicIotaOp(t_inter1, dim=0)"
        )

    def test_mlir(self, flat_ir):
        print(str(flat_ir.to_mlir()))
        helper.check_mlir(
            flat_ir.to_mlir(),
            """
            module {
                func.func @main() -> tensor<?x?xf32> {
                    %c = stablehlo.constant dense<[2, 3]> : tensor<2xi32>
                    %0 = stablehlo.dynamic_iota %c, dim = 0 : (tensor<2xi32>) -> tensor<?x?xf32>
                    return %0 : tensor<?x?xf32>
                }
            }
            """,
        )
