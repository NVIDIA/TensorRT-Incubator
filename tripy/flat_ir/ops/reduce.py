from dataclasses import dataclass
from typing import List

from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp
from tripy.backend.mlir.utils import get_mlir_dtype


@dataclass
class ReduceOp(BaseFIROp):
    """
    Operation to reduce a Tensor
    """

    reduce_mode: str
    reduce_dims: List[int]

    def __init__(self, origin_layer, inputs, outputs, reduce_dims, reduce_mode):
        super().__init__(origin_layer, inputs, outputs)
        self.reduce_dims = list(reduce_dims)
        self.reduce_mode = reduce_mode

    # TODO(#87): Reuse flat ir ops
    def _get_reduce_func(self):
        if self.reduce_mode == "sum":
            return stablehlo.AddOp
        elif self.reduce_mode == "max":
            return stablehlo.MaxOp
        else:
            raise NotImplementedError()

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        reduce = stablehlo.ReduceOp(
            result=[out_type],
            inputs=[operands[0]],
            init_values=[operands[1]],
            dimensions=self.reduce_dims,
        )

        input_dtype = get_mlir_dtype(self.inputs[0].dtype)
        reduce_arg_type = ir.RankedTensorType.get(
            [],
            input_dtype,
        )
        reduce_block = ir.Block.create_at_start(reduce.regions[0], [reduce_arg_type, reduce_arg_type])
        reduce_func = self._get_reduce_func()
        with ir.InsertionPoint(reduce_block):
            out = reduce_func(*reduce_block.arguments)
            stablehlo.ReturnOp([out])

        return [reduce]
