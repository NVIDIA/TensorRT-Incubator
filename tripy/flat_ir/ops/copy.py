from typing import List

from mlir import ir

from tripy.flat_ir.ops.base import BaseFIROp
import tripy.common
from dataclasses import dataclass


@dataclass
class CopyOp(BaseFIROp):
    """
    Operation to copy a tensor to another device
    """

    target: tripy.common.device

    def __init__(self, origin_layer, inputs, outputs, target):
        super().__init__(origin_layer, inputs, outputs)
        self.target = target

    def to_mlir(self, operands: List) -> List:
        from mlir.dialects import bufferization, tensor

        assert len(operands) == 1 and len(self.inputs) == 1, "Copy should have exactly one input!"
        mem_space_str = "device" if self.target.kind == "gpu" else "host_pinned"
        mem_space_attr = ir.Attribute.parse(f"#executor.memory_type<{mem_space_str}>")
        if self.inputs[0].to_mlir().has_static_shape:
            alloc_tensor = bufferization.alloc_tensor(
                self.inputs[0].to_mlir(), [], memory_space=mem_space_attr, copy=operands[0]
            )
            return [alloc_tensor]
        else:
            alloc_tensor = bufferization.alloc_tensor(
                operands[0].results[0].type, [], memory_space=mem_space_attr, copy=operands[0]
            )
            cast_tensor = tensor.cast(self.outputs[0].to_mlir(), alloc_tensor)
            return [cast_tensor]
