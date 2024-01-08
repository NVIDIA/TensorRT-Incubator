from typing import List

from mlir import ir

from tripy.flat_ir.ops.base import BaseFIROp
from tripy.util.util import get_flat_tensor_info


class CopyOp(BaseFIROp):
    """
    Operation to copy a tensor to another device
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert "target" in kwargs
        self.target = kwargs.get("target")

    def to_flat_ir_str(self, input_names, output_names) -> str:
        assert len(output_names) == 1, "CompareOp should have exactly one output!"
        return f"{output_names[0]} = {self.__class__.__name__} copy={get_flat_tensor_info(input_names[0], self.inputs[0])}, target={self.target.kind}:{self.target.index})"

    def to_mlir(self, operands: List) -> List:
        from mlir.dialects import bufferization

        assert len(operands) == 1 and len(self.inputs) == 1, "Copy should have exactly one input!"
        mem_space_str = "device" if self.target.kind == "gpu" else "host_pinned"
        mem_space_attr = ir.Attribute.parse(f"#executor.memory_type<{mem_space_str}>")
        dst_tensor = bufferization.alloc_tensor(
            self.inputs[0].to_mlir(), [], memory_space=mem_space_attr, copy=operands[0]
        )
        return [dst_tensor]
