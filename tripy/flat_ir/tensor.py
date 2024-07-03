import contextlib
import copy
from dataclasses import dataclass
from typing import Any, List, Optional

from tripy import utils
from tripy.common.types import ShapeInfo
from tripy.frontend.dim import dynamic_dim

_BUILD_CONTEXT: List[Any] = []


@dataclass(repr=False)
class FlatIRTensor:
    """
    Represents a single tensor in the FlatIR
    """

    name: str
    stack_info: utils.StackInfo
    dtype: "tripy.common.dtype"
    device: "tripy.common.device"
    rank: int
    producer: "BaseFlatIROp" = None
    reason_details: Optional[List[Any]] = None
    shape: ShapeInfo = None
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
        try:
            global _BUILD_CONTEXT
            _BUILD_CONTEXT.append(ctx)
            yield
        finally:
            _BUILD_CONTEXT.pop()

    def to_mlir(self):
        from tripy.backend.mlir import utils as mlir_utils

        shape = (
            utils.to_dims([dynamic_dim(-1) for i in range(self.rank)])
            if (self.shape is None or self.shape == [])
            else self.shape
        )
        return mlir_utils.make_mlir_tensor(shape, self.dtype)

    @staticmethod
    def build(
        dtype: "tripy.common.dtype", device: "tripy.common.device", rank: int, reason_details: List[Any], shape=None
    ) -> "FlatIRTensor":
        return FlatIRTensor(
            name=None,
            # Include code from the caller of this function up, and not just user code
            # since this is an intermediate tensor created within tripy.
            stack_info=utils.get_stack_info(include_code_index=1),
            shape=shape,
            dtype=dtype,
            device=device,
            rank=rank,
            producer=None,
            reason_details=reason_details,
            reason_context=copy.copy(_BUILD_CONTEXT),
        )

    def __str__(self) -> str:
        def str_from_dim(dim: "dynamic_dim"):
            return ("?" if dim.is_dynamic_dim() else str(dim)) + ","

        shape = self.shape if (self.shape != [] and self.shape is not None) else [dynamic_dim(-1)] * self.rank
        return (
            f"{self.name}: [rank=({self.rank}), "
            + (f"shape=({' '.join(map(str_from_dim, shape))}), ")
            + (f"dtype=({self.dtype.name}), " if self.dtype is not None else "")
            + f"loc=({self.device})]"
        )
