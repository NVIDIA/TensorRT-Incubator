#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List, Set, Union, Optional

from tripy import utils
from tripy.utils import Result


@dataclass(repr=False)
class BaseTraceOp(abc.ABC):
    inputs: List["TraceTensor"]
    """The inputs of this layer"""

    outputs: List["TraceTensor"]
    """The outputs of this layer"""

    @classmethod
    def build_internal(
        cls, inputs: List["TraceTensor"], outputs: List["TraceTensor"], *args, **kwargs
    ) -> "BaseTraceOp":
        """
        Builds a Trace operation and binds it to the provided input and output trace tensors.

        *args and **kwargs are passed along to the trace operation's constructor.
        """
        from tripy.frontend.trace.tensor import TraceTensor

        assert all(isinstance(tensor, TraceTensor) for tensor in inputs + outputs)

        op = cls(inputs, outputs, *args, **kwargs)
        for out in op.outputs:
            out.producer = op

        op.infer_dtypes()
        op.infer_rank()
        op.infer_devices()
        return op

    @classmethod
    def build(cls, inputs: List["Tensor"], *args, num_outputs=1, **kwargs) -> Union["Tensor", List["Tensor"]]:
        """
        Builds a trace operation and binds its inputs to the trace tensors corresponding to the
        frontend tensors provided in `inputs` and creates `num_outputs` new frontend tensors for the
        outputs, whose trace tensors are bound to the outputs of the trace operation.

        *args and **kwargs are passed along to the trace operation's constructor.

        `num_outputs=1` is treated as a special case that will return the output tensor directly instead
        of returning a list of output tensors.
        """

        from tripy.common.exception import raise_error
        from tripy.frontend.shape import Shape
        from tripy.frontend.tensor import Tensor

        # NOTE: If you change the stack depth where the tensors are constructed, update STACK_DEPTH_OF_BUILD in
        # the Tensor constructor!
        outputs = [Tensor(None) for _ in range(num_outputs)]

        inp_trace_tensors = [inp.trace_tensor for inp in inputs]
        out_trace_tensors = [out.trace_tensor for out in outputs]
        op = cls.build_internal(inp_trace_tensors, out_trace_tensors, *args, **kwargs)

        # wrap shape outputs if necessary
        res = op.infer_shape_output_idxs(inputs)
        if not res:
            custom_err = "" if not res.error_details else " Further information: " + "\n".join(res.error_details)
            shape_arg_idxs = [i for i in range(len(inputs)) if isinstance(inputs[i], Shape)]
            shape_arg_msg = "none" if len(shape_arg_idxs) == 0 else ", ".join(map(str, shape_arg_idxs))
            raise_error(
                f"Error processing shape inputs in operator {cls.__name__}{custom_err}\n(Shape input indices: {shape_arg_msg}.)"
            )
        # for shape outputs, we infer the length
        if len(res.value) != 0:
            inferred_lengths = op.infer_len()
        for idx in res.value:
            outputs[idx] = Shape(outputs[idx])
            if inferred_lengths[idx] is not None:
                out_trace_tensors[idx].shape = [inferred_lengths[idx]]

        if num_outputs == 1:
            return outputs[0]
        return outputs

    def infer_shape_output_idxs(self, inputs: List["Tensor"]) -> Result:
        """
        Given the operator's inputs, this method returns a `Result` containing a list of the operator's output indices
        that should be wrapped in `tp.Shape`.

        By default, this will wrap all the outputs in `tp.Shape` if all the inputs are `tp.Shape`s and not wrap any otherwise,
        treating it as an error if the inputs are inconsistent.

        Operators may override this method to enforce a different rule, such as expecting only some inputs to be `tp.Shape`
        and others not to be.

        To avoid duplicating error-checking logic, this method should return an error value only if the error
        would not otherwise be caught in the operator implementation (e.g., if the result is not `int32` or rank 1,
        the Shape constructor would report an error anyway, so this check should not also check for that).

        Args:
            inputs: The operator's (front-end `Tensor`) inputs

        Returns:
            A `Result` containing, if successful, a list of indices of outputs that should be converted to `tp.Shape`.
        """
        from tripy.frontend.shape import Shape

        is_shape = lambda t: isinstance(t, Shape)

        if any(map(is_shape, inputs)):
            if all(map(is_shape, inputs)):
                return Result.ok(list(range(len(self.outputs))))
            return Result.err(["Either all inputs must be tp.Shape or all must be tp.Tensor."])
        return Result.ok([])

    def infer_len(self) -> List[Optional[int]]:
        """
        Infers the length of all `tp.Shape` outputs. This is, essentially, the "shape" of the shape.
        Returns `None` for outputs that are not `tp.Shape`s or whose length (shape) cannot be inferred.

        Returns:
            A list of inferred lengths for outputs that are `tp.Shape`s.
        """
        return [None for _ in self.outputs]

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

    def infer_rank(self):
        """
        Infers and updates rank for the output of the operation.
        """
        assert (
            self.inputs and len(self.outputs) == 1
        ), "Default implementation cannot handle cases where there are no inputs, multiple outputs."
        # Max for all input ranks is done to account for rank broadcasting.
        self.outputs[0].rank = max(inp.rank for inp in self.inputs)

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
