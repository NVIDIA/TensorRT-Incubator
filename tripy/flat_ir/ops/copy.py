from typing import List

from mlir import ir

from tripy.flat_ir.ops.base import BaseFIROp


class CopyOp(BaseFIROp):
    """
    Operation to copy a tensor to another device
    """

    def __init__(self, origin_layer, inputs, outputs, target):
        super().__init__(origin_layer, inputs, outputs)
        self.target = target

    def to_mlir(self, operands: List) -> List:
        from mlir.dialects import bufferization

        assert len(operands) == 1 and len(self.inputs) == 1, "Copy should have exactly one input!"
        mem_space_str = "device" if self.target.kind == "gpu" else "host_pinned"
        mem_space_attr = ir.Attribute.parse(f"#executor.memory_type<{mem_space_str}>")
        dst_tensor = bufferization.alloc_tensor(
            self.inputs[0].to_mlir(), [], memory_space=mem_space_attr, copy=operands[0]
        )
        return [dst_tensor]
