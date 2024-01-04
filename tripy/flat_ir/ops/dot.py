from typing import List

from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class DotOp(BaseFIROp):
    """
    Operation to compute generic dot product of two tensors.
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        default_dict = {"lhs": [], "rhs": []}
        self.contracting_dim = kwargs.get("contracting_dim", default_dict)
        self.batching_dim = kwargs.get("batching_dim", default_dict)

    def to_flat_ir_str(self, input_names, output_names) -> str:
        assert len(output_names) == 1, f"{self.__class__.__name__} should have exactly one output!"
        return f"{output_names[0]} = {self.__class__.__name__} {' '.join(input_names)}"

    def to_mlir(self, operands: List) -> List:
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
