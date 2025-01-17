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

import copy
from typing import List, Sequence, Set

from nvtripy import utils
from nvtripy.common.exception import raise_error
from nvtripy.common.shape_bounds import ShapeBounds
from nvtripy.frontend.trace.ops import Storage
from nvtripy.frontend.trace.tensor import TraceTensor
from nvtripy.frontend.utils import topological_sort
from nvtripy.logging import logger


class Trace:
    """
    A flattened representation of a computation graph expressed by one or more Tensors.
    """

    def __init__(
        self,
        tensors: Sequence[TraceTensor],
        inputs: Sequence[TraceTensor] = [],
        shapes: Sequence[ShapeBounds] = None,
    ) -> None:
        """
        Args:
            tensors: The output TraceTensor(s) to evaluate.
            inputs: Input TraceTensor(s).
            shapes: The shape profile, consisting of min, opt, and max shapes for each input tensor.
                    Must be in the same order as `inputs`.
        """
        self.ops: List["BaseTraceOp"] = []
        self.inputs: List[TraceTensor] = inputs
        self.outputs: List[TraceTensor] = tensors
        self.shapes = shapes

        exprs = [tensor.producer for tensor in tensors]

        input_op_ids = set(id(inp.producer) for inp in inputs)
        seen_op_ids: Set[int] = set()

        # Check all tensors for duplicate names. We currently rely on tensor names being
        # unique in the trace/flatIR. We could potentially change this in the future to
        # automatically make names unique instead of complaining to the user, but it's better
        # for traceability if we use the names set by the user/frontend.
        _tensor_map = {}

        def check_name(tensor):
            if tensor.name in _tensor_map and (_tensor_map[tensor.name] is not tensor):
                raise_error(
                    f"Found distinct tensors with the same name: '{tensor.name}'.",
                    details=["Tensor: ", tensor, "has the same name as another tensor: ", _tensor_map[tensor.name]],
                )
            _tensor_map[tensor.name] = tensor

        while exprs:
            head = exprs.pop(0)

            if id(head) in seen_op_ids:
                continue
            seen_op_ids.add(id(head))

            for io in head.inputs + head.outputs:
                check_name(io)

            if id(head) not in input_op_ids:
                # not as an input
                self.ops.append(head)
                exprs.extend([inp.producer for inp in head.inputs])

        # Reverse the order of the layers so they are topologically sorted
        self.ops = topological_sort(self.ops)

        logger.trace(lambda: f"{self}\n")

    def __str__(self) -> str:
        layer_strs: List[str] = []
        if self.shapes:
            layer_strs.append("input shapes:")
            for shape in self.shapes:
                layer_strs.append(f"    {str(shape)}")

        if len(self.inputs):
            layer_strs.append("inputs:")
        for inp in self.inputs:
            layer_strs.append(f"    {str(inp)}")
        for op in self.ops:
            layer_strs.append(str(op))
        layer_strs.append("outputs:")
        for out in self.outputs:
            layer_strs.append(f"    {str(out)}")
        return "\n".join(layer_strs)

    @staticmethod
    def _collect_storage_tensors(trace_tensor):
        visited = set()
        inputs = []

        def dfs(trace_tensor):
            if id(trace_tensor) in visited:
                return
            visited.add(id(trace_tensor))

            producer = trace_tensor.producer
            if isinstance(producer, Storage) and utils.utils.should_lift_storage_op_as_input(producer.shape):
                inputs.append(trace_tensor)
            else:
                for inp in producer.inputs:
                    dfs(inp)

        dfs(trace_tensor)
        return inputs

    def to_flat_ir(self):
        from nvtripy.flat_ir.flat_ir import FlatIR

        flat_ir = FlatIR(shapes=self.shapes)
        # Assign shapes to static shape arguments to ease translation and optimizations during the lowering to MLIR.
        if self.shapes:
            for input, shape_bounds in zip(self.inputs, self.shapes):
                if shape_bounds.is_static():
                    assert all(
                        s >= 0 for s in shape_bounds.min
                    ), f"shape bounds expected to be >= 0, got {shape_bounds.min}"
                    input.shape = list(shape_bounds.min)

        flat_ir.inputs = [flat_ir.register_tensor(inp.to_flat_ir()) for inp in self.inputs]
        flat_ir.outputs = [flat_ir.register_tensor(out.to_flat_ir()) for out in self.outputs]

        for op in self.ops:
            inputs = [flat_ir.register_tensor(inp.to_flat_ir()) for inp in op.inputs]
            outputs = [flat_ir.register_tensor(out.to_flat_ir()) for out in op.outputs]
            # Pass shallow copies of inputs/outputs so that the op is free to modify them
            op.to_flat_ir(copy.copy(inputs), copy.copy(outputs))
            flat_ir.integrate_subgraph(inputs, outputs)

        logger.flat_ir(lambda: f"{flat_ir}\n")
        return flat_ir
