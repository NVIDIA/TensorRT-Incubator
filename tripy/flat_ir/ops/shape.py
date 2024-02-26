from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp
from tripy.backend.mlir import utils as mlir_utils
from tripy import int32


class ShapeOp(BaseFlatIROp):
    def to_mlir(self, operands):

        inp = operands[0]

        if not (isinstance(operands[0], ir.OpResult) or isinstance(operands[0], ir.BlockArgument)):
            inp = inp.result

        assert inp.type.rank > 0, "ShapeOp should only be used when input tensor has rank > 0."

        sliced_dims = [None] * inp.type.rank
        # Loop and slice all indicies, concat to yield shape tensor.
        # Remove use of tensorrt dialect and use shape dialect. #80 will fix this.
        for i in range(inp.type.rank):
            broadcast_dim_attr = ir.IntegerAttr.get(
                type=ir.IntegerType.get_signless(64),
                value=i,
            )

            dim_size = stablehlo.get_dimension_size(inp, dimension=broadcast_dim_attr)
            out_type = ir.RankedTensorType.get([1], mlir_utils.get_mlir_dtype(int32))
            sliced_dims[i] = stablehlo.ReshapeOp(result=out_type, operand=dim_size)

        concatenate_dim = ir.IntegerAttr.get(
            type=ir.IntegerType.get_signless(64),
            value=0,
        )
        output = stablehlo.concatenate(sliced_dims, dimension=concatenate_dim)
        return [output]
