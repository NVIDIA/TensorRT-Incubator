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

import numbers
from textwrap import indent
from typing import Any, List, Optional, Union

# Import ops to populate the registry before we define our Tensor class
import nvtripy.frontend.ops
from nvtripy import export, utils
from nvtripy.backend.mlir import memref
from nvtripy.common import datatype
from nvtripy.common import device as tp_device
from nvtripy.common.exception import raise_error, str_from_stack_info
from nvtripy.frontend.ops._registry import TENSOR_METHOD_REGISTRY
from nvtripy.logging.logger import logger
from nvtripy.trace.ops.constant import Constant


class TensorMeta(type):
    def __new__(cls, name, bases, dct):
        new = type.__new__(cls, name, bases, dct)

        # We only register methods with the Tensor class. Derived classes
        # will inherit these methods normally. If we register for derived classes too
        # we run the risk of overwriting overridden methods.
        if name == "Tensor":
            # Add methods specified by individual ops to this class.
            for method_name, method_impl in TENSOR_METHOD_REGISTRY.items():
                setattr(new, method_name, method_impl)

        return new


@export.public_api(
    document_under="tensor/index.rst",
    autodoc_options=[
        ":special-members:",
        ":exclude-members: __init__, __repr__, __weakref__, __dlpack__, __dlpack_device__",
    ],
    # Skip function registry dispatch for __init__ to avoid runtime overheads.
    bypass_dispatch=["__init__"],
)
class Tensor(metaclass=TensorMeta):
    """
    A tensor is a multi-dimensional array that contains elements of a uniform data type.
    """

    # This field communicates to NumPy that it should allow our right-side operator overloads (e.g. __radd__) to take
    # precedence over its own left-side overloads (e.g. __add__). This will ensure that an expression of the form
    # `<np_array> <binary_op> Tensor` will return a Tensor and not a NumPy array.
    __array_priority__ = 10000

    def __init__(
        self,
        data: Any,
        dtype: Optional["nvtripy.dtype"] = None,
        device: Optional["nvtripy.device"] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            data: The data with which to initialize the tensor.
                For types that support the DLPack protocol, copying data is avoided if possible.

            dtype: The data type of the tensor.
            device: The device on which to allocate the tensor.
                If the provided data is not on this device, it will be copied.
                By default, the tensor will be allocated on the same device as the `data` argument.

            name: The name of the tensor. If provided, this must be a unique string.

        .. code-block:: python
            :linenos:

            tensor = tp.Tensor([1.0, 2.0, 3.0], dtype=tp.float32)
        """
        # We use None internally but users should not be permitted to do it
        assert data is not None, "Data argument to Tensor must not be None"
        if isinstance(data, Tensor):
            raise_error(
                "Cannot initialize Tensor with another Tensor.", [f"Note: `data` argument was defined here:", data]
            )

        self._stack_info = utils.stack_info.StackInfo([])

        # We include a fast path for the case where we're initializing a Tensor from a DLPack
        # tensor with no modifications. This usually happens at runtime if we're interoperating
        # with other frameworks (e.g. multiple executables with framework glue code in between).
        # In that case, we don't want to add any overheads like fetching stack information.
        if hasattr(data, "__dlpack__") and device is None and dtype is None:
            constant = Constant(data, device=device, dtype=dtype)
            self.trace_tensor = constant.outputs[0]
            self.trace_tensor.name = utils.utils.default(name, self.trace_tensor.name)
            self.trace_tensor.stack_info = self._stack_info
            return

        # Small optimization for scalars to avoid unnecessary casts:
        if isinstance(data, numbers.Number) and dtype is not None:
            if issubclass(dtype, datatype.floating):
                data = float(data)
            elif issubclass(dtype, datatype.integer):
                data = int(data)
            elif dtype == datatype.bool:
                data = bool(data)

        constant = Constant(data, device=device, dtype=dtype)
        self.trace_tensor = constant.outputs[0]
        self.trace_tensor.name = utils.utils.default(name, self.trace_tensor.name)
        self.stack_info = utils.stack_info.get_stack_info(include_code_index=1)

        # Preserve the device after casting if necessary (otherwise cast will always copy to GPU):
        device = utils.utils.default(device, self.device)

        # Cast/copy if necessary:
        casted_copied_tensor = self
        if dtype is not None and dtype != casted_copied_tensor.dtype:
            from nvtripy.frontend.ops.cast import cast

            casted_copied_tensor = cast(casted_copied_tensor, dtype=dtype)

        # We do not check trace_tensor.device, since that will always be GPU
        # (Constants always generate outputs in GPU memory).
        if device is not None and device != casted_copied_tensor.device:
            # Copy to the new device
            from nvtripy.frontend.ops.copy import copy

            casted_copied_tensor = copy(casted_copied_tensor, device=device)

        # We must evaluate the new tensor prior to assigning self.trace_tensor or we could
        # end up in an infinite loop since the input *and* output of cast/copy would both
        # point to this frontend tensor.
        casted_copied_tensor._eval_for_internal_methods()
        self.trace_tensor = casted_copied_tensor.trace_tensor

    # Left undocumented because these should only be used internally.
    @classmethod
    def from_trace_tensor(cls, trace_tensor, include_code_index=2):
        instance = cls.__new__(cls)
        instance.trace_tensor = trace_tensor
        instance.stack_info = utils.stack_info.get_stack_info(include_code_index=include_code_index)
        return instance

    def __getattr__(self, name: str):
        import nvtripy as tp
        from nvtripy.common.exception import search_for_missing_attr

        look_in = [(tp, "nvtripy")]
        search_for_missing_attr("nvtripy.Tensor", name, look_in)

    @property
    def trace_tensor(self):
        return self._trace_tensor

    @trace_tensor.setter
    def trace_tensor(self, new_trace_tensor):
        self._trace_tensor = new_trace_tensor
        self._trace_tensor.frontend_tensor = self

    @property
    def name(self):
        return self.trace_tensor.name

    @name.setter
    def name(self, new_name):
        self.trace_tensor.name = new_name

    @property
    def stack_info(self):
        return self._stack_info

    @stack_info.setter
    def stack_info(self, new_stack_info):
        self._stack_info = new_stack_info
        self.trace_tensor.stack_info = new_stack_info

    @property
    def dtype(self):
        return self.trace_tensor.dtype

    @property
    def rank(self):
        return self.trace_tensor.rank

    @property
    def ndim(self):
        return self.trace_tensor.rank

    @property
    def device(self):
        # For constants, we want to report where the data currently resides.
        # Note that on evaluation, it will always be copied to the device.
        if isinstance(self.trace_tensor.producer, Constant):
            return self.trace_tensor.producer.device
        return self.trace_tensor.device

    def eval(self) -> "nvtripy.Tensor":
        """
        Immediately evaluates this tensor. By default, tensors are evaluated lazily.

        .. note:: The evaluated tensor will always be in **GPU memory**.

        Returns:
            The evaluated tensor.

        .. code-block:: python
            :linenos:

            import time

            start = time.perf_counter()
            tensor = tp.ones((3, 3))
            init_time = time.perf_counter()
            tensor.eval()
            eval_time = time.perf_counter()

            print(f"Tensor init_time took: {(init_time - start)  * 1000.0:.3f} ms")
            print(f"Tensor evaluation took: {(eval_time - init_time)  * 1000.0:.3f} ms")
        """
        if isinstance(self.trace_tensor.producer, Constant) and self.trace_tensor.producer.device.kind == "gpu":
            # Exit early if the tensor has already been evaluated.
            # We can only do this for Constants that are already on the GPU, since otherwise
            # we need to evaluate in order to perform a copy from the host to the device.
            # This happens before the imports below so we don't incur extra overhead.
            return self

        from nvtripy.backend.api.executable import Executable
        from nvtripy.backend.mlir.compiler import Compiler
        from nvtripy.trace.trace import Trace
        from nvtripy.backend.api.input_info import InputInfo

        trace = Trace([self.trace_tensor])

        # TensorRT requires all constants to start in host memory, so if there's anything on GPU
        # already, we pull it out into an input.
        inputs = []
        for op in trace.ops:
            if isinstance(op, Constant) and op.device.kind == "gpu":
                inputs.append(op.outputs[0].frontend_tensor)

        if inputs:
            trace.trace([self.trace_tensor], [inp.trace_tensor for inp in inputs])

        compiler = Compiler(trt_builder_opt_level=0)
        mlir = trace.to_mlir()
        arg_names = [f"arg{i}" for i in range(len(inputs))]

        executable = Executable(
            compiler.compile(mlir, trace=trace),
            arg_names=arg_names,
            return_single_tensor_as_sequence=False,
            # Input shapes should always be statically known since only GPU constants are turned into inputs.
            # We need to manually fetch the trace_tensor shape since the `.shape` op will always create a subgraph
            # for compile tracers (i.e. it will fetch a non-constant shape if we eval while compiling).
            input_infos={
                name: InputInfo(list(map(int, inp.trace_tensor.shape)), inp.dtype)
                for name, inp in zip(arg_names, inputs)
            },
        )
        data = executable(*inputs).trace_tensor.producer.data

        # Upon computing the value of this tensor, we switch it to have a `Constant`
        # parameter so that it does not need to be computed again.
        constant = Constant(data)
        # Need to carry forward `is_compile_tracer`:
        constant.outputs[0].is_compile_tracer = self.trace_tensor.is_compile_tracer

        # Rebind this tensor, but be sure to preserve stack information:
        self.trace_tensor = constant.outputs[0]
        self.trace_tensor.stack_info = self.stack_info

        self.trace_tensor.eval_stack_info = utils.stack_info.get_stack_info()
        if self.trace_tensor.is_compile_tracer:
            logger.warning(
                f"Tensor was evaluated while compiling which may cause unexpected behavior in the executable.\n"
                f"For example, this could cause values to be baked into the executable or dynamic shapes to become static.\n"
                f"If the result of the evaluation is not being used by other operations, you can safely ignore this warning.",
                mode="once",
            )
            logger.warning(
                f"Note: Tensor was evaluated while compiling here: {str_from_stack_info(self.trace_tensor.eval_stack_info)}",
                mode="once",
            )
        return self

    # Special version of eval() that skips evaluation for Constants in host memory.
    # This should only be used if the method does not care where the data resides.
    def _eval_for_internal_methods(self):
        if not isinstance(self.trace_tensor.producer, Constant):
            self.eval()

    def tolist(self) -> Union[List, numbers.Number]:
        """
        Returns the tensor as a nested list. If the tensor is a scalar, returns a python number.

        Returns:
            The tensor represented as a nested list or a python number.

        .. code-block:: python
            :caption: Ranked tensor
            :linenos:

            # doc: print-locals tensor_list
            tensor = tp.ones((2, 2))
            tensor_list = tensor.tolist()

            assert tensor_list == np.ones((2, 2), dtype=np.float32).tolist()

        .. code-block:: python
            :caption: Scalar
            :linenos:

            # doc: print-locals tensor_scalar
            tensor = tp.Tensor(2.0, dtype=tp.float32)
            tensor_scalar = tensor.tolist()

            assert tensor_scalar == 2.0
        """
        self._eval_for_internal_methods()
        data_memref = self.trace_tensor.producer.data
        if self.dtype not in (
            datatype.float32,
            datatype.int8,
            datatype.int32,
            datatype.int64,
            datatype.bool,
        ):
            from nvtripy.frontend.ops.cast import cast

            cast_tensor = cast(Tensor(data_memref), datatype.float32)
            cast_tensor._eval_for_internal_methods()
            data_memref = cast_tensor.trace_tensor.producer.data
        return memref.tolist(data_memref)

    def __iter__(self):
        raise TypeError("Iterating over tensors is not supported")

    def __repr__(self) -> str:
        from nvtripy.frontend.utils import pretty_print

        data_list = self.tolist()

        assert isinstance(self.trace_tensor.producer, Constant)
        data_shape = self.trace_tensor.producer.shape

        arr_str = pretty_print(data_list, data_shape)
        indentation = ""
        sep = ""
        if len(data_shape) > 1 and any(dim > 1 for dim in data_shape):
            indentation = " " * 4
            sep = "\n"
        return (
            f"tensor({sep}"
            f"{indent(arr_str, prefix=indentation)}, {sep}"
            f"{indent(f'dtype={self.dtype}, loc={self.device}, shape={data_shape}', prefix=indentation)}"
            f")"
        )

    # Since the underlying data is an MemRefValue we reuse their __dlpack__() and __dlpack_device__() methods
    def __dlpack__(self, stream: Any = None):
        self._eval_for_internal_methods()
        return self.trace_tensor.producer.data.__dlpack__()

    def __dlpack_device__(self):
        self._eval_for_internal_methods()
        return self.trace_tensor.producer.data.__dlpack_device__()

    def __bool__(self):
        data = self.tolist()

        assert isinstance(self.trace_tensor.producer, Constant)
        if any(dim != 1 for dim in self.trace_tensor.producer.shape):
            raise_error(
                "Boolean value of a Tensor with more than one value is ambiguous",
                [f"Note: tensor shape was: {self.trace_tensor.producer.shape}"],
            )

        # Unwrap, since the item could be nested within a list. Without unwrapping, `[[[0]]]` returns True, when this should return False.
        for _ in range(self.rank):
            data = data[0]
        return bool(data)
