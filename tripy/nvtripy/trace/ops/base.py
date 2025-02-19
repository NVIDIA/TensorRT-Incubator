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
from dataclasses import dataclass, field
from typing import List, Set

from nvtripy import utils
from nvtripy.trace.tensor import TraceTensor

_COUNT = 0


def _get_unique_name():
    global _COUNT

    name = f"t{_COUNT}"
    _COUNT += 1
    return name


@dataclass(repr=False)
class BaseTraceOp(abc.ABC):
    """
    Abstract base class for trace operations in the computational graph.

    This class represents a node in the trace graph, with inputs and outputs
    as TraceTensor objects.
    """

    inputs: List["TraceTensor"]
    """The input tensors of this operation"""

    outputs: List["TraceTensor"] = field(init=False)
    """The output tensors of this operation"""

    def __post_init__(self):
        is_compile_tracer = any(inp.is_compile_tracer for inp in self.inputs)
        self.outputs = [
            TraceTensor(_get_unique_name(), producer=self, is_compile_tracer=is_compile_tracer)
            for _ in range(self.get_num_outputs())
        ]

        self.infer_dtypes()
        self.infer_rank()
        self.infer_devices()

    def get_num_outputs(self) -> int:
        """
        The number of output produced by this trace operation.
        """
        return 1

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
            for field in utils.utils.get_dataclass_fields(self, BaseTraceOp)
            if field.name not in skip_fields
        ]
        return f"{self.outputs[0].name} = {self.__class__.__name__.lower()}({', '.join([inp.name for inp in self.inputs] + args)})"

    def __repr__(self) -> str:
        # This is a hack to prevent printing the entire stack info when we print this.
        return str(self)
