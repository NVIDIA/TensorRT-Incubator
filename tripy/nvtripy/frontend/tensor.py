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

from textwrap import indent
from typing import Any, Optional

import mlir_tensorrt.runtime.api as runtime

# Import ops to populate the registry before we define our Tensor class
import nvtripy.frontend.ops
from nvtripy import export, utils
from nvtripy.backend.mlir import memref
from nvtripy.common import datatype
from nvtripy.common.exception import raise_error, str_from_stack_info
from nvtripy.frontend.ops._registry import TENSOR_METHOD_REGISTRY
from nvtripy.logging.logger import logger
from nvtripy.trace.ops.storage import Storage


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
        fetch_stack_info: bool = True,
    ) -> None:
        """
        Args:
            data: The data with which to initialize the tensor.
            dtype: The data type of the tensor.
            device: The device on which to allocate the tensor.
            name: The name of the tensor. If provided, this must be a unique string.
            fetch_stack_info: Whether to fetch stack information for the tensor.
                Stack information allows Tripy to generate much higher quality error
                messages at the cost of a small overhead when initializing the tensor.

        .. code-block:: python
            :linenos:

            tensor = tp.Tensor([1.0, 2.0, 3.0], dtype=tp.float32)
        """
        # We use None internally but users should not be permitted to do it
        assert data is not None, "Data argument to Tensor must not be None"
        self._stack_info = utils.stack_info.StackInfo([])

        storage = Storage(data, device=device if not hasattr(data, "__dlpack__") else None)
        self.trace_tensor = storage.outputs[0]
        self.trace_tensor.name = utils.utils.default(name, self.trace_tensor.name)
        if fetch_stack_info:
            # TODO (pranavm): Figure out the right stack depth
            self.stack_info = utils.stack_info.get_stack_info(include_code_index=1)

        # TODO(#155): Remove this hack:
        self.trace_tensor.device = utils.utils.default(device, self.trace_tensor.device)

        # Explicit cast if necessary
        # TODO(#155): Add copy as well when host allocation is fixed
        if dtype is not None and dtype != self.trace_tensor.dtype:
            from nvtripy.frontend.ops.cast import cast

            self.trace_tensor = cast(self, dtype=dtype).trace_tensor

    # Left undocumented because these should only be used internally.
    @classmethod
    def from_trace_tensor(cls, trace_tensor, include_code_index=2):
        instance = cls.__new__(cls)
        instance.trace_tensor = trace_tensor
        # TODO (pranavm): Figure out what stack depth to use here?
        instance.stack_info = utils.stack_info.get_stack_info(include_code_index=include_code_index)
        return instance

    # Faster constructor that bypasses things like function registry type checks and fetching stack info.
    @staticmethod
    def fast_init(data: Any):
        instance = Tensor.__new__(Tensor)
        storage = Storage(data)
        instance.trace_tensor = storage.outputs[0]
        instance.stack_info = utils.stack_info.StackInfo([])
        return instance

    def __getattr__(self, name: str):
        import nvtripy as tp
        from nvtripy.common.exception import search_for_missing_attr

        look_in = [(tp, "nvtripy")]
        search_for_missing_attr("nvtripy.Tensor", name, look_in)

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
        return self.trace_tensor.device

    def eval(self) -> runtime.MemRefValue:
        if isinstance(self.trace_tensor.producer, Storage):
            # Exit early if the tensor has already been evaluated.
            # This happens before the imports below so we don't incur extra overhead.
            return self.trace_tensor.producer.data

        from nvtripy.backend.api.executable import Executable
        from nvtripy.backend.mlir.compiler import Compiler
        from nvtripy.frontend.cache import global_cache
        from nvtripy.trace.trace import Trace

        # Collect inputs
        inputs = Trace._collect_storage_tensors(self.trace_tensor)

        trace = Trace([self.trace_tensor], inputs=inputs)
        output_devices = [out.device for out in trace.outputs]

        executable = global_cache.get(trace, devices=output_devices)
        if executable is None:
            flat_ir = trace.to_flat_ir()
            mlir = flat_ir.to_mlir()

            compiler = Compiler(trt_builder_opt_level=0)
            executable = compiler.compile(mlir, flat_ir=flat_ir)

            global_cache.set(trace, executable=executable, devices=output_devices)

        data = Executable(executable, [])().eval()

        # Upon computing the value of this tensor, we switch it to have a `Storage`
        # parameter so that it does not need to be computed again.
        storage = Storage(data)
        # Need to carry forward `is_compile_tracer`:
        storage.outputs[0].is_compile_tracer = self.trace_tensor.is_compile_tracer

        # Rebind this tensor, but be sure to preserve stack information:
        self.trace_tensor = storage.outputs[0]
        self.trace_tensor.stack_info = self.stack_info

        # TODO(#155): Remove this hack of overriding the device type.
        self.trace_tensor.device = output_devices[0]

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

        return data

    def tolist(self):
        data_memref = self.eval()
        if self.dtype not in (
            datatype.float32,
            datatype.int8,
            datatype.int32,
            datatype.int64,
            datatype.bool,
        ):
            from nvtripy.frontend.ops.cast import cast

            data_memref = cast(Tensor(data_memref), datatype.float32).eval()
        return memref.tolist(data_memref)

    def __iter__(self):
        raise TypeError("Iterating over tensors is not supported")

    def __repr__(self) -> str:
        from nvtripy.frontend.utils import pretty_print

        data_list = self.tolist()

        assert isinstance(self.trace_tensor.producer, Storage)
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
        return self.eval().__dlpack__()

    def __dlpack_device__(self):
        return self.eval().__dlpack_device__()

    def __bool__(self):
        data = self.tolist()

        assert isinstance(self.trace_tensor.producer, Storage)
        if any(dim != 1 for dim in self.trace_tensor.producer.shape):
            raise_error(
                "Boolean value of a Tensor with more than one value is ambiguous",
                [f"Note: tensor shape was: {self.trace_tensor.producer.shape}"],
            )

        # Unwrap, since the item could be nested within a list. Without unwrapping, `[[[0]]]` returns True, when this should return False.
        for _ in range(self.rank):
            data = data[0]
        return bool(data)
