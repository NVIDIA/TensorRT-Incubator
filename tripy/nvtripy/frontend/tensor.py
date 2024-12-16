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
import nvtripy.frontend.trace.ops
from nvtripy import export, utils
from nvtripy.backend.mlir import memref
from nvtripy.common import datatype
from nvtripy.common.exception import raise_error, str_from_stack_info
from nvtripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from nvtripy.frontend.trace.ops import Storage
from nvtripy.frontend.trace.tensor import TraceTensor
from nvtripy.logging.logger import logger
from nvtripy.utils.stack_info import StackInfo


# We include code for everything above the `BaseTraceOp.build` function, which is called at most
# this many stack frames above the constructor.
STACK_DEPTH_OF_BUILD = 5


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

    _COUNT = 0

    # This field communicates to NumPy that it should allow our right-side operator overloads (e.g. __radd__) to take
    # precedence over its own left-side overloads (e.g. __add__). This will ensure that an expression of the form
    # `<np_array> <binary_op> Tensor` will return a Tensor and not a NumPy array.
    __array_priority__ = 10000

    @classmethod
    def _get_unique_name(cls):
        name = f"t{cls._COUNT}"
        cls._COUNT += 1
        return name

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
            :caption: Example

            tensor = tp.Tensor([1.0, 2.0, 3.0], dtype=tp.float32)
        """
        # We use None internally but users should not be permitted to do it
        assert data is not None, "Data argument to Tensor must not be None"
        Tensor.raw_init(self, data, dtype, device, name, fetch_stack_info)

    # Left undocumented because this should only be used internally.
    # Produces a new instance of a Tensor but avoids calling into the function registry, unlike the normal constructor.
    @staticmethod
    def create_directly(
        data: Any,
        dtype: Optional["nvtripy.dtype"] = None,
        device: Optional["nvtripy.device"] = None,
        name: Optional[str] = None,
        fetch_stack_info: bool = True,
    ):
        instance = Tensor.__new__(Tensor)
        Tensor.raw_init(instance, data, dtype, device, name, fetch_stack_info)
        return instance

    # No docstring because this should be used only internally. Handles the logic for initializing a new instance.
    # We separate this from __init__ because __init__ calls into the registry and rejects None values, which we use internally.
    @staticmethod
    def raw_init(
        instance: Any,
        data: Any,
        dtype: Optional["nvtripy.dtype"] = None,
        device: Optional["nvtripy.device"] = None,
        name: Optional[str] = None,
        fetch_stack_info: bool = True,
    ):
        stack_info = StackInfo([])
        if fetch_stack_info:
            stack_info = utils.get_stack_info(include_code_index=STACK_DEPTH_OF_BUILD)

        name = name if name is not None else Tensor._get_unique_name()

        instance.trace_tensor = TraceTensor(name, stack_info, dtype=None, device=device, producer=None, shape=None)

        # Note: It is important that we are able to call the Tensor constructor with no arguments
        # since this is used internally.
        if data is None:
            return

        if hasattr(data, "__dlpack__"):
            if not isinstance(data, runtime.MemRefValue):
                data = memref.create_memref_view(data)
            Storage.build_internal([], [instance.trace_tensor], data)
        else:
            Storage.build_internal([], [instance.trace_tensor], data, device)
        # TODO(#155): Remove this hack:
        instance.trace_tensor.device = utils.default(device, instance.trace_tensor.device)

        # Explicit cast if necessary
        # TODO(#155): Add copy as well when host allocation is fixed
        if dtype is not None and dtype != instance.trace_tensor.dtype:
            from nvtripy.frontend.trace.ops.cast import cast

            instance.trace_tensor = cast(instance, dtype=dtype).trace_tensor

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
        return self.trace_tensor.stack_info

    @stack_info.setter
    def stack_info(self, new_stack_info):
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

        from nvtripy.backend.mlir.compiler import Compiler
        from nvtripy.backend.mlir.executor import Executor
        from nvtripy.frontend.trace import Trace

        trace = Trace([self])
        flat_ir = trace.to_flat_ir()
        mlir = flat_ir.to_mlir()

        compiler = Compiler(trt_builder_opt_level=0)
        executable = compiler.compile(mlir, flat_ir=flat_ir)
        executor = Executor(executable)
        # Upon computing the value of this tensor, we switch it to have a `Storage`
        # parameter so that it does not need to be computed again.
        data = executor.execute([out.device for out in flat_ir.outputs])
        executor.stream.synchronize()
        assert len(data) == 1, "Expects only one output from mlir_tensorrt.compiler executor"
        data = data[0]

        Storage.build_internal([], [self.trace_tensor], data)
        # TODO(#155): Remove this hack of overriding the device type.
        self.trace_tensor.device = flat_ir.outputs[0].device

        self.trace_tensor.eval_stack_info = utils.get_stack_info()
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
            from nvtripy.frontend.trace.ops.cast import cast

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
