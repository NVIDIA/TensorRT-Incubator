#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvtripy.common.device import device
from nvtripy.trace.tensor import TraceTensor

_COUNT = 0


def _get_unique_name():
    global _COUNT

    name = f"%t{_COUNT}"
    _COUNT += 1
    return name


@dataclass(repr=False)
class TraceOp(abc.ABC):
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

        # TODO (#577): Support multiple devices here:
        # All operations in TRT create outputs on the GPU:
        for out in self.outputs:
            out.device = device.fast_init("gpu", 0)

    @abc.abstractmethod
    def to_mlir(self, inputs: List["ir.Operation"], outputs: List["ir.RankedTensorType"]) -> List["ir.Operation"]:
        """
        Generates MLIR operations for the operation.

        Args:
            inputs: The input MLIR operations.
            outputs: The tensor types of the outputs.

        Returns:
            The output MLIR operations, which may be the same as `outputs`, or may be newly created
            outputs, in which case `outputs` will be discarded.
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
        skip_fields = self.str_skip_fields()
        args = [
            f"{field.name}={getattr(self, field.name)}"
            for field in utils.utils.get_dataclass_fields(self, TraceOp)
            if field.name not in skip_fields
        ]

        out_types = f"{', '.join(f'tensor{out.type_descriptor()}' for out in self.outputs)}"
        if len(self.outputs) > 1:
            out_types = f"({out_types})"

        return (
            f"{', '.join(out.name for out in self.outputs)} = {utils.utils.pascal_to_snake_case(self.__class__.__name__)}"
            f"({', '.join([inp.name + f' : tensor{inp.type_descriptor()}' for inp in self.inputs] + args)})"
            f" : {out_types}"
        )

    def __repr__(self) -> str:
        # This is a hack to prevent printing the entire stack info when we print this.
        return str(self)
