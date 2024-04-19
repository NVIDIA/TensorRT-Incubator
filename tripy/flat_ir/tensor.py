import contextlib
import copy
from dataclasses import dataclass
from typing import Any, List, Optional

from tripy import utils
from tripy.common.types import ShapeInfo

_BUILD_CONTEXT: List[Any] = None


@dataclass(repr=False)
class FlatIRTensor:
    """
    Represents a single tensor in the FlatIR
    """

    name: str
    stack_info: utils.StackInfo
    shape: ShapeInfo
    dtype: "tripy.common.dtype"
    device: "tripy.common.device"
    producer: "BaseFlatIROp" = None
    reason_details: Optional[List[Any]] = None
    """
    Describes why this tensor was created.
    This is required for any intermediate tensors created by Tripy,
    but optional for tensors that can be traced back to user tensors.
    It should complete the sentence: "This operation was added in order to...".
    """
    reason_context: Optional[List[Any]] = None

    @contextlib.contextmanager
    @staticmethod
    def context(ctx: List[Any]):
        # TODO (#165): Make this support hierarchical contexts. Doing so should be fairly easy -
        # we just need to make _BUILD_CONTEXT a list of lists which we push/pop from.
        try:
            global _BUILD_CONTEXT
            _BUILD_CONTEXT = ctx
            yield
        finally:
            _BUILD_CONTEXT = None

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
            shape=shape,
            dtype=dtype,
            device=device,
            producer=None,
            reason_details=reason_details,
            reason_context=copy.copy(_BUILD_CONTEXT),
        )

    def __str__(self) -> str:
        def str_from_dim(dim: "dynamic_dim"):
            return ("?" if dim.is_dynamic_dim() else str(dim)) + ","

        shape = f"{' '.join(map(str_from_dim, self.shape))}"
        return (
            f"{self.name}: [shape=({shape}), "
            + (f"dtype=({self.dtype.name}), " if self.dtype is not None else "")
            + f"loc=({self.device})]"
        )
