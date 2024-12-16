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

import abc
from dataclasses import dataclass
from typing import List, Optional, Set, Union

from nvtripy import utils


@dataclass(repr=False)
class BaseTraceOp(abc.ABC):
    """
    Abstract base class for trace operations in the computational graph.

    This class represents a node in the trace graph, with inputs and outputs
    as TraceTensor objects.
    """

    inputs: List["TraceTensor"]
    """The input tensors of this operation"""

    outputs: List["TraceTensor"]
    """The output tensors of this operation"""

    @classmethod
    def build_internal(
        cls, inputs: List["TraceTensor"], outputs: List["TraceTensor"], *args, **kwargs
    ) -> "BaseTraceOp":
        """
        Builds a Trace operation and binds it to the provided input and output trace tensors.

        *args and **kwargs are passed along to the trace operation's constructor.
        """
        op = cls(inputs, outputs, *args, **kwargs)

        is_compile_tracer = any(inp.is_compile_tracer for inp in inputs)
        for out in op.outputs:
            out.producer = op
            out.is_compile_tracer |= is_compile_tracer

        op.infer_dtypes()
        op.infer_rank()
        op.infer_devices()
        return op

    @classmethod
    def build(
        cls, inputs: List["Tensor"], *args, num_outputs=1, always_cast_to_dimension_size=False, **kwargs
    ) -> Union["Tensor", List["Tensor"]]:
        """
        Builds a trace operation and binds its inputs to the trace tensors corresponding to the
        frontend tensors provided in `inputs` and creates `num_outputs` new frontend tensors for the
        outputs, whose trace tensors are bound to the outputs of the trace operation.

        *args and **kwargs are passed along to the trace operation's constructor.

        `num_outputs=1` is treated as a special case that will return the output tensor directly instead
        of returning a list of output tensors.
        """

        from nvtripy.common.datatype import int32
        from nvtripy.frontend.dimension_size import DimensionSize
        from nvtripy.frontend.tensor import Tensor

        # NOTE: If you change the stack depth where the tensors are constructed, update STACK_DEPTH_OF_BUILD in
        # the Tensor constructor!
        outputs = [Tensor.create_directly(None) for _ in range(num_outputs)]

        inp_trace_tensors = [inp.trace_tensor for inp in inputs]
        out_trace_tensors = [out.trace_tensor for out in outputs]
        cls.build_internal(inp_trace_tensors, out_trace_tensors, *args, **kwargs)

        # Operations that operate on only DimensionSize inputs will always yield a DimensionSize.
        # For any mixed operations, DimensionSize must be casted up to Tensor.
        all_inputs_are_dimension_size = all(isinstance(inp, DimensionSize) for inp in inputs)
        for index, out in enumerate(outputs):
            if always_cast_to_dimension_size or (
                all_inputs_are_dimension_size and out.dtype == int32 and out.rank == 0
            ):
                dim_size = DimensionSize.create_directly(None)
                dim_size.trace_tensor = out.trace_tensor
                dim_size.stack_info = out.stack_info
                outputs[index] = dim_size

        if num_outputs == 1:
            return outputs[0]
        return outputs

    @abc.abstractmethod
    def infer_rank(self):
        """
        Infers the rank of the output.
        """
        ...

    def infer_dtypes(self):
        """
        Infers dtypes for the operation and updates output tensor dtypes accordingly.
        """
        assert self.inputs, "Default implementation cannot handle cases where there are no inputs. Please override."
        assert (
            len(self.outputs) == 1
        ), f"Default implementation expects exactly one output, but got {len(self.outputs)}. Please override."
        assert all(
            inp.dtype == self.inputs[0].dtype for inp in self.inputs
        ), f"Default implementation cannot handle cases where inputs have different dtypes, but got {[inp.dtype for inp in self.inputs]}. Please override."

        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_devices(self):
        """
        Infers devices for the operation and updates output tensor devices accordingly.
        """
        assert (
            self.inputs and len(self.outputs) == 1 and all(inp.device == self.inputs[0].device for inp in self.inputs)
        ), "Default implementation cannot handle cases where there are no inputs, multiple outputs, or multiple inputs with different devices. Please override."
        self.outputs[0].device = self.inputs[0].device

    @abc.abstractmethod
    def to_flat_ir(self, inputs: List["FlatIRTensor"], outputs: List["FlatIRTensor"]):
        """
        Generates a FlatIR subgraph for the operation and binds it to the specified
        inputs and outputs.

        Args:
            inputs: The inputs to the subgraph.
            outputs: The outputs of the subgraph.
        """
        ...

    def str_skip_fields(self) -> Set[str]:
        """
        Returns names of dataclass fields to skip when generating a string representation of the op.
        """
        return set()

    def __str__(self) -> str:
        """
        Returns a Trace string representation of the operation.

        Returns:
            The Trace string representation of the operation.
        """
        assert len(self.outputs) == 1, "Base class implementation only works for single output operations!"

        skip_fields = self.str_skip_fields()
        args = [
            f"{field.name}={getattr(self, field.name)}"
            for field in utils.get_dataclass_fields(self, BaseTraceOp)
            if field.name not in skip_fields
        ]
        return f"{self.outputs[0].name} = {self.__class__.__name__.lower()}({', '.join([inp.name for inp in self.inputs] + args)})"

    def __repr__(self) -> str:
        # This is a hack to prevent printing the entire stack info when we print this.
        return str(self)
