import abc
from dataclasses import dataclass
from typing import List, Set, Union

from tripy import utils


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
            out.shape = []

        op.infer_dtypes()
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

        from tripy.frontend.tensor import Tensor

        outputs = [Tensor(None) for _ in range(num_outputs)]

        inp_trace_tensors = [inp.trace_tensor for inp in inputs]
        out_trace_tensors = [out.trace_tensor for out in outputs]
        cls.build_internal(inp_trace_tensors, out_trace_tensors, *args, **kwargs)

        if num_outputs == 1:
            return outputs[0]
        return outputs

    @abc.abstractmethod
    def infer_shapes(self):
        """
        Infers shapes for the operation and updates output tensor shapes accordingly.
        """
        ...

    def infer_dtypes(self):
        """
        Infers dtypes for the operation and updates output tensor dtypes accordingly.
        """
        assert (
            self.inputs and len(self.outputs) == 1 and all(inp.dtype == self.inputs[0].dtype for inp in self.inputs)
        ), "Default implementation cannot handle cases where there are no inputs, multiple outputs, or multiple inputs with different data types. Please override."
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
