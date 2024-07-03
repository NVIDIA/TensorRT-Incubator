from dataclasses import dataclass

from mlir_tensorrt.compiler import ir

import tripy.common
from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class CopyOp(BaseFlatIROp):

    target: tripy.common.device

    def to_mlir(self, operands):
        from mlir_tensorrt.compiler.dialects import bufferization, tensor, arith

        assert len(operands) == 1 and len(self.inputs) == 1, "Copy should have exactly one input!"
        mem_space_str = "device" if self.target.kind == "gpu" else "host_pinned"
        mem_space_attr = ir.Attribute.parse(f"#executor.memory_type<{mem_space_str}>")
        if self.inputs[0].to_mlir().has_static_shape:
            alloc_tensor = bufferization.alloc_tensor(
                self.inputs[0].to_mlir(), [], memory_space=mem_space_attr, copy=operands[0]
            )
            return [alloc_tensor]
        else:
            inp_type = operands[0].type if hasattr(operands[0], "type") else operands[0].result.type
            sliced_dims = []
            # Loop and slice all indices, concat to yield shape tensor.
            for i in range(inp_type.rank):
                if inp_type.is_dynamic_dim(i):
                    idx = arith.ConstantOp.create_index(i)
                    dim = tensor.DimOp(operands[0], idx)
                    sliced_dims.append(dim)

            alloc_tensor = bufferization.alloc_tensor(inp_type, sliced_dims, memory_space=mem_space_attr)
            result_tensor = bufferization.materialize_in_destination(inp_type, operands[0], alloc_tensor)
            cast_tensor = tensor.cast(self.outputs[0].to_mlir(), result_tensor)

            return [cast_tensor]
