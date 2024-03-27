from typing import Dict, List, Sequence

from tripy import utils
from tripy.logging import logger
from tripy.frontend import Tensor
from tripy.frontend.trace import Trace
from tripy.frontend.trace.ops import Storage


def get_trace_signature(trace: Trace) -> int:
    """
    Returns the signature of a trace. Two traces with the same signature are structurally
    equivalent for the purposes of executable caching.

    NOTE: This is an overly strict check, so traces that are isomorphic yet have small differences
        (e.g. swapped order of outputs) may have different signatures, triggerring a recompilation.
    """
    from tripy.frontend.trace.ops import BaseTraceOp

    def get_op_signature(op: BaseTraceOp) -> int:
        # For ops, we don't actually care about the I/O tensors except in how they're connected
        # (i.e. graph structure). Only the trace-level I/O tensors matter, and those are checked separately.

        if isinstance(op, Storage):
            # Don't consider shapes/data for storage tensors
            return utils.md5(op.dtype, op.device)

        return utils.md5(
            [getattr(op, field.name) for field in utils.get_dataclass_fields(op, BaseTraceOp)],
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

    def get_tensors_signatures(tensors: Sequence[Tensor]) -> List[int]:
        # For now, we only consider the structural ID in the tensor signature.
        # Data types/shapes are checked as part of the CachedExecutable instead.
        return [utils.md5(tensor_structural_ids[tensor.name]) for tensor in tensors]

    inp_signatures = get_tensors_signatures(trace.inputs)
    logger.verbose(f"Input signatures: {inp_signatures}")

    op_signatures = [
        # Consider structure when generating op signatures.
        (get_op_signature(op), *[tensor_structural_ids[tensor.name] for tensor in op.inputs + op.outputs])
        for op in trace.ops
    ]
    logger.verbose(f"Op signatures: {op_signatures}")

    out_signatures = get_tensors_signatures(trace.outputs)
    logger.verbose(f"Output signatures: {out_signatures}")

    return utils.md5(
        *inp_signatures,
        *op_signatures,
        *out_signatures,
    )
