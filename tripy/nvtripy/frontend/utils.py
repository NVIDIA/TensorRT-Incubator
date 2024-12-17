#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import deque
from typing import List, Union

from nvtripy.common.exception import raise_error
from nvtripy.flat_ir.ops import BaseFlatIROp


def tensor_from_shape_like(arg: "nvtripy.ShapeLike") -> "nvtripy.Tensor":
    from nvtripy.common.datatype import int32
    from nvtripy.frontend.dimension_size import DimensionSize
    from nvtripy.frontend.tensor import Tensor
    from nvtripy.frontend.trace.ops.concatenate import concatenate
    from nvtripy.frontend.trace.ops.reshape import Reshape

    if not arg:
        return Tensor.create_directly([], dtype=int32)

    concat_tensors = []

    # We accumulate integers so we can create just a single tensor for each contiguous
    # sequence of integers.
    int_buffer = []

    def empty_buffer():
        if not int_buffer:
            return

        concat_tensors.append(Tensor.create_directly(int_buffer, dtype=int32))
        int_buffer.clear()

    for elem in arg:
        if isinstance(elem, DimensionSize):
            empty_buffer()
            # NOTE: We cannot use the reshape API here since it would lead to an
            # infinite loop when attempting to convert the shape input to a tensor.
            concat_tensors.append(Reshape.build([elem, Tensor.create_directly([1])], 1))
        else:
            int_buffer.append(elem)

    empty_buffer()

    out = concatenate(concat_tensors, dim=0)
    # We must set the shape of the shape tensor here since otherwise we will not be able
    # to infer ranks in the frontend. Note that the reshape operations above will not result
    # in a tensor with known shapes even though the new shape is actually known.
    out.trace_tensor.shape = [len(arg)]
    return out


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


# Processes a `dim` (i.e. axis) argument related to a tensor.
# If the dimension is negative, this will convert it to the corresponding positive index.
def process_dim(dim: int, input_rank: int) -> int:
    new_dim = dim
    if dim < 0:
        new_dim = input_rank + dim

    if new_dim < 0 or new_dim >= input_rank:
        raise_error(
            "Dimension argument is out of bounds.",
            [
                f"Note: provided dimension was: {dim}, while the tensor has a rank of: {input_rank}.\n"
                f"Dimension should be in the half-open interval: [{-input_rank}, {input_rank})."
            ],
        )
    return new_dim


def pretty_print(data_list, shape, threshold=1000, linewidth=10, edgeitems=3):
    """
    Returns a pretty-print string of list format data.
    """

    def _data_str(data, summarize, linewidth, edgeitems, indent=0):
        if isinstance(data, (float, int)):
            return str(data)

        if len(data) == 0 or isinstance(data[0], (float, int)):
            if summarize and len(data) > 2 * edgeitems:
                data_lines = [data[:edgeitems] + [" ..."] + data[-edgeitems:]]
            else:
                data_lines = [data[i : i + linewidth] for i in range(0, len(data), linewidth)]
            lines = [", ".join([f"{e:.4f}" if isinstance(e, float) else str(e) for e in line]) for line in data_lines]
            return "[" + ("," + "\n" + " " * (indent + 1)).join(lines) + "]"

        if summarize and len(data) > 2 * edgeitems:
            slices = (
                [_data_str(data[i], summarize, linewidth, edgeitems, indent + 1) for i in range(0, edgeitems)]
                + ["..."]
                + [
                    _data_str(data[i], summarize, linewidth, edgeitems, indent + 1)
                    for i in range(len(data) - edgeitems, len(data))
                ]
            )
        else:
            slices = [_data_str(data[i], summarize, linewidth, edgeitems, indent + 1) for i in range(0, len(data))]

        tensor_str = ("," + "\n" * (max(len(shape) - indent - 1, 1)) + " " * (indent + 1)).join(slices)
        return "[" + tensor_str + "]"

    numel = 1
    for d in shape:
        numel *= d
    summarize = numel > threshold
    return _data_str(data_list, summarize, linewidth, edgeitems)
