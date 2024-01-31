from typing import Any, Dict, Sequence

from tripy import utils
from tripy.common.types import ShapeInfo
from tripy.frontend import Tensor
from tripy.frontend.trace import Trace
from tripy.utils.json import Decoder, Encoder


# HACK (#109): This is a temporary class which we need in order to convey output information
# to MLIR-TRT. Eventually, it should just call back into Tripy when it needs memory allocated.
class TensorInfo:
    def __init__(self, shape: ShapeInfo, dtype: "tripy.dtype", device: "tripy.device"):
        self.shape = shape
        self.dtype = dtype
        self.device = device


@Encoder.register(TensorInfo)
def encode(tensor_info: TensorInfo) -> Dict[str, Any]:
    return {"shape": tensor_info.shape, "dtype": tensor_info.dtype, "device": tensor_info.device}


@Decoder.register(TensorInfo)
def decode(dct: Dict[str, Any]) -> TensorInfo:
    return TensorInfo(dct["shape"], dct["dtype"], dct["device"])


def get_tensor_info(tensors):
    return [TensorInfo(tensor.shape, tensor.dtype, tensor.device) for tensor in tensors]


def get_trace_signature(trace: Trace) -> int:
    """
    Returns the signature of a trace. Two traces with the same signature are structurally
    equivalent for the purposes of executable caching.

    NOTE: This is an overly strict check, so traces that are isomorphic yet have small differences
        (e.g. swapped order of outputs) may have different signatures, triggerring a recompilation.
    """
    from tripy.frontend.ops import BaseOperator

    def get_op_signature(op: BaseOperator) -> int:
        # For ops, we don't actually care about the I/O tensors except in how they're connected
        # (i.e. graph structure). Only the trace-level I/O tensors matter, and those are checked separately.
        from tripy.frontend.ops import Storage

        if isinstance(op, Storage):
            return utils.md5(op.shape, op.dtype, op.device)

        return utils.md5(
            [getattr(op, field.name) for field in utils.get_dataclass_fields(op, BaseOperator)],
        )

    # In addition to the tensors/ops in this trace, we also need to consider the structure.
    # To do that, we'll assign numerical IDs to each tensor that are independent of the tensors
    # themselves and use those to augment the signatures of the tensors/ops.
    #
    # Maps tensor names (which are globally unique) to their structural IDs (which are locally unique).
    tensor_structural_ids: Dict[int, int] = {}

    def set_tensor_sid(tensors: Sequence[Tensor]):
        for tensor in tensors:
            if tensor.name not in tensor_structural_ids:
                tensor_structural_ids[tensor.name] = len(tensor_structural_ids)

    set_tensor_sid(trace.inputs)
    for op in trace.ops:
        set_tensor_sid(op.inputs)
        set_tensor_sid(op.outputs)
    set_tensor_sid(trace.outputs)

    def get_tensors_signature(tensors: Sequence[Tensor]) -> int:
        # For now, we only consider the structural ID in the tensor signature.
        # Data types/shapes are checked as part of the CachedExecutable instead.
        return utils.md5([tensor_structural_ids[tensor.name] for tensor in tensors])

    return utils.md5(
        get_tensors_signature(trace.inputs),
        *[
            # Consider structure when generating op signatures.
            (get_op_signature(op), *[tensor_structural_ids[tensor.name] for tensor in op.inputs + op.outputs])
            for op in trace.ops
        ],
        get_tensors_signature(trace.outputs),
    )
