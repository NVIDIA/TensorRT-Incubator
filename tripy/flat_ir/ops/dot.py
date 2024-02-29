from dataclasses import dataclass
from typing import Dict, List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class DotOp(BaseFlatIROp):

    contracting_dim: Dict[str, List[int]]
    batching_dim: Dict[str, List[int]]

    def to_mlir(self, operands):
        # dot_general spec: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general
        out_type = self.outputs[0].to_mlir()

        attr = stablehlo.DotDimensionNumbers.get(
            lhs_batching_dimensions=self.batching_dim["lhs"],
            rhs_batching_dimensions=self.batching_dim["rhs"],
            lhs_contracting_dimensions=self.contracting_dim["lhs"],
            rhs_contracting_dimensions=self.contracting_dim["rhs"],
        )

        dot_out = stablehlo.dot_general(result=out_type, lhs=operands[0], rhs=operands[1], dot_dimension_numbers=attr)
        return [dot_out]
