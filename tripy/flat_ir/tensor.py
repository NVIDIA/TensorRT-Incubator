from dataclasses import dataclass

from tripy import utils
from tripy.common.types import ShapeInfo
from tripy.frontend.trace.tensor import TraceTensor

from typing import Optional, List, Any


@dataclass(repr=False)
class FlatIRTensor(TraceTensor):
    """
    Represents a single tensor in the FlatIR
    """

    reason_details: Optional[List[Any]] = None
    """
    Describes why this tensor was created.
    This is required for any intermediate tensors created by Tripy,
    but optional for tensors that can be traced back to user tensors.
    It should complete the sentence: "This operation was added in order to...".
    """

    def to_mlir(self):
        from tripy.backend.mlir import utils as mlir_utils

        return mlir_utils.make_mlir_tensor(self.shape, self.dtype)

    @staticmethod
    def build(
        shape: ShapeInfo,
        dtype: "tripy.common.dtype",
        device: "tripy.common.device",
        reason_details: List[Any],
    ) -> "FlatIRTensor":
        return FlatIRTensor(
            name=None,
            # Include code from the caller of this function up, and not just user code
            # since this is an intermediate tensor created within tripy.
            stack_info=utils.get_stack_info(include_code_index=1),
            producer=None,
            shape=shape,
            dtype=dtype,
            device=device,
            reason_details=reason_details,
        )
