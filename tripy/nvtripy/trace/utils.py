from collections import deque
from typing import List, Union

from nvtripy.flat_ir.ops import BaseFlatIROp


def topological_sort(ops: List[Union["BaseTraceOp", BaseFlatIROp]]) -> List[Union["BaseTraceOp", BaseFlatIROp]]:
    """
    This utility to topologically sort a graph that can be a Trace or a FlatIR graph.
    """
    stack = deque()
    visited_layer_ids = set()
    result_set = set()
    result = list()
    id_ops = set(id(op) for op in ops)

    for op in ops:
        if id(op) not in visited_layer_ids:
            stack.append((op, False))

            while stack:
                current_op, is_processed = stack.pop()
                if id(current_op) in result_set:
                    continue
                if is_processed:
                    result.append(current_op)
                    result_set.add(id(current_op))
                    continue

                visited_layer_ids.add(id(current_op))
                stack.append((current_op, True))

                for ip in reversed(current_op.inputs):
                    if (
                        ip.producer is not None
                        and id(ip.producer) not in visited_layer_ids
                        and id(ip.producer) in id_ops
                    ):
                        stack.append((ip.producer, False))

    assert len(ops) == len(result), f"Num original ops {len(ops)}, got num {len(result)}"
    return result
