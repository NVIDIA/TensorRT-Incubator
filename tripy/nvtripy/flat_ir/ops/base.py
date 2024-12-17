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
from typing import List, Set, Tuple


from nvtripy import utils


@dataclass(repr=False)
class BaseFlatIROp(abc.ABC):
    """
    Represents a single layer in the FlatIR.
    """

    inputs: List["FlatIRTensor"]
    """The inputs of this operation"""

    outputs: List["FlatIRTensor"]
    """The outputs of this operation"""

    # Trace input/output names are populated by FlatIR.integrate_subgraph().
    trace_input_names: List[str]
    """The names of the input trace tensors of the FlatIR subgraph this operation is part of"""

    trace_output_names: List[str]
    """The names of the output trace tensors of the FlatIR subgraph this operation is part of"""

    @classmethod
    def build(cls, inputs: List["FlatIRTensor"], outputs: List["FlatIRTensor"], *args, **kwargs) -> "BaseFlatIROp":
        from nvtripy.flat_ir.tensor import FlatIRTensor

        assert all(isinstance(tensor, FlatIRTensor) for tensor in inputs + outputs)

        op = cls(inputs, outputs, [], [], *args, **kwargs)
        for out in op.outputs:
            out.producer = op
        return op

    @abc.abstractmethod
    def to_mlir(self, operands: List["ir.Operation"]) -> List["ir.Operation"]:
        """
        Generates MLIR HLO operations for the operation.

        Args:
            operands: The input MLIR HLO operations.

        Returns:
            The generated MLIR HLO operations.
        """
        ...

    def str_skip_fields(self) -> Set[str]:
        """
        Returns names of dataclass fields to skip when generating a string representation of the op.
        """
        return set()

    def __str__(self) -> str:
        """
        Returns a FlatIR string representation of the operation.

        Returns:
            The FlatIR string representation of the operation.
        """
        outputs_str = (
            f"{str(self.outputs[0])}" if len(self.outputs) == 1 else ", ".join([str(out) for out in self.outputs])
        )
        skip_fields = self.str_skip_fields()
        args = [
            f"{field.name}={repr(getattr(self, field.name))}"
            for field in utils.get_dataclass_fields(self, BaseFlatIROp)
            if field.name not in skip_fields
        ]
        return f"{outputs_str} = {self._op_name()}({', '.join([str(inp.name) for inp in self.inputs] + args)})"

    def __repr__(self) -> str:
        # This is a hack to prevent printing the entire stack info when we print this.
        return str(self)

    def _op_name(self) -> str:
        """
        Returns the human readable name of this operation.

        Returns:
            The name of this operation.
        """
        return self.__class__.__name__
