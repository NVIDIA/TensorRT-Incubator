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


from tripy import utils


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
        from tripy.flat_ir.tensor import FlatIRTensor

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


class FlatIRFunction(abc.ABC):
    """Represents a function in the Flat IR."""

    def __init__(self, name: str, inputs: List["FlatIRTensor"], outputs: List["FlatIRTensor"]):
        """Initialize a FlatIRFunction."""
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.ops: List[BaseFlatIROp] = []
        self.traced_ir_ops: List[BaseFlatIROp] = []  # Used for flat ir function deduplication
        self.trace_input_names = None
        self.trace_output_names = None
        self.caller_replacements = []

    def clone_with_new_io(
        self, new_inputs: List["FlatIRTensor"], new_outputs: List["FlatIRTensor"]
    ) -> "FlatIRFunction":
        """
        Create a clone of the function with new inputs and outputs.
        """
        new_func = FlatIRFunction(self.name, new_inputs, new_outputs)
        new_func.ops = self.ops
        new_func.trace_input_names = self.trace_input_names
        new_func.trace_output_names = self.trace_output_names
        return new_func

    def set_caller_inputs(self, inputs: List["FlatIRTensor"]) -> None:
        for callee_input, caller_input in zip(self.inputs, inputs):
            callee_input.caller_tensor = caller_input

    def get_caller_inputs(self) -> List["FlatIRTensor"]:
        return [getattr(inp, "caller_tensor") for inp in self.inputs]

    def get_caller_outputs(self) -> List["FlatIRTensor"]:
        return [getattr(out, "caller_tensor") for out in self.outputs]

    def add_op(self, op: BaseFlatIROp) -> None:
        """
        Add an operation to the function ops. `trace_ir_ops` are used only for function deduplication.
        """
        self.ops.append(op)
        self.traced_ir_ops.append(op)

    def __str__(self) -> str:
        """Generate a string representation of the function."""
        function_signature = [
            f"function {self.name}(",
            *[f"    {inp}" for inp in self.inputs],
            ") -> (",
            *[f"    {out}" for out in self.outputs],
            ") {",
        ]
        function_body = [f"    {op}" for op in self.ops]
        function_return = [f"    return {', '.join(out.name for out in self.outputs)}", "}"]
        return "\n".join(function_signature + function_body + function_return)

    def __repr__(self) -> str:
        """Generate a concise string representation of the function."""
        return f"<FlatIRFunction '{self.name}' with {len(self.inputs)} inputs, {len(self.outputs)} outputs, and {len(self.ops)} ops>"

    def is_structurally_equivalent(self, other: "FlatIRFunction") -> bool:
        """Check if two FlatIRFunction objects are structurally equivalent."""

        def tensor_signature(tensor: "FlatIRTensor") -> tuple:
            return (tensor.shape, tensor.dtype, tensor.rank, tensor.device)

        def op_signature(op: "BaseFlatIROp") -> tuple:
            from tripy.flat_ir.ops import ConstantOp

            return (
                type(op),
                [tensor_signature(t) for t in op.inputs],
                [tensor_signature(t) for t in op.outputs],
                tuple(
                    getattr(op, field.name)
                    for field in utils.get_dataclass_fields(op, BaseFlatIROp)
                    if field.name not in ("inputs", "outputs")
                ),
            )

        # Check high-level structure
        if (
            len(self.inputs) != len(other.inputs)
            or len(self.outputs) != len(other.outputs)
            or len(self.traced_ir_ops) != len(other.traced_ir_ops)
        ):
            return False

        # Check input and output signatures
        if any(
            tensor_signature(t1) != tensor_signature(t2)
            for t1, t2 in zip(self.inputs + self.outputs, other.inputs + other.outputs)
        ):
            return False

        # Check ops and its input and output signatures
        return all(op_signature(op1) == op_signature(op2) for op1, op2 in zip(self.traced_ir_ops, other.traced_ir_ops))
