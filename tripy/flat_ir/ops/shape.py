from dataclasses import dataclass

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.backend.mlir import utils as mlir_utils
from tripy.flat_ir.ops.base import BaseFlatIROp
from tripy.common.array import Array
from tripy.common.device import device


@dataclass(repr=False)
class ShapeOp(BaseFlatIROp):
    def to_mlir(self, operands):
        from tripy.common.datatype import int32

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

            data = Array([1], dtype=int32, device=device("cpu"))
            attr = ir.DenseElementsAttr.get(
                array=data.memref_value,
                type=mlir_utils.get_mlir_dtype(int32),
                shape=[
                    1,
                ],
            )

            output_shape = stablehlo.ConstantOp(attr)
            sliced_dims[i] = stablehlo.dynamic_reshape(result=out_type, operand=dim_size, output_shape=output_shape)

        concatenate_dim = ir.IntegerAttr.get(
            type=ir.IntegerType.get_signless(64),
            value=0,
        )
        output = stablehlo.concatenate(sliced_dims, dimension=concatenate_dim)
        return [output]
