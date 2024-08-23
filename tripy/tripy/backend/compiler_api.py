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

import atexit
import base64
import inspect
import numbers
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Union

import mlir_tensorrt.runtime.api as runtime

from tripy import export, utils
from tripy.backend.mlir import Compiler as MLIRCompiler
from tripy.backend.mlir import Executor
from tripy.backend.mlir import utils as mlir_utils
from tripy.common.exception import raise_error
from tripy.common.shape_bounds import ShapeBounds
from tripy.frontend import Tensor, Trace
from tripy.utils import json as json_utils
from tripy.backend.mlir.utils import MLIRRuntimeClient


@export.public_api(document_under="compiler")
class Stream:
    """
    Represents a CUDA stream that can be used to manage concurrent operations.

    This class is a wrapper around the underlying stream object, allowing management of CUDA streams.

    .. code-block:: python
        :linenos:
        :caption: Use streams in execution

        linear = tp.Linear(2, 3)
        compiler = tp.Compiler(linear)

        compiled_linear = compiler.compile(tp.InputInfo((2, 2), dtype=tp.float32))
        with tp.Stream():
            a = tp.ones((2, 2), dtype=tp.float32)
            out = compiled_linear(a)

        assert cp.array_equal(cp.from_dlpack(out), cp.from_dlpack(linear(a)))
    """

    _active_stream = None
    _default_stream = None

    def __init__(self):
        self._stream = MLIRRuntimeClient().create_stream()

    @classmethod
    def default_stream(cls):
        """Get the default stream, create it if it doesn't exist."""
        if cls._default_stream is None:
            cls._default_stream = cls.__new__(cls)
            cls._default_stream._stream = MLIRRuntimeClient().create_stream()

        return cls._default_stream

    def synchronize(self):
        """Synchronize the stream, blocking until all operations in this stream are complete."""
        self._stream.sync()

    @classmethod
    def get_current_stream(cls):
        return Stream._active_stream if Stream._active_stream is not None else Stream._default_stream

    def __enter__(self):
        Stream._active_stream = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.synchronize()
        # TODO: expose methods from mlir-tensorrt python bindings to destroy stream
        if self == Stream._active_stream:
            Stream._active_stream = Stream._default_stream

    def __eq__(self, other):
        if not isinstance(other, Stream):
            return False

        if not (hasattr(self, "_stream") and hasattr(other, "_stream")):
            return False

        return self._stream == other._stream

    def __repr__(self):
        return f"<Stream(id={id(self)}, default={self == Stream._default_stream})>"

    @classmethod
    def cleanup_default_stream(cls):
        if cls._default_stream:
            # TODO: expose methods from mlir-tensorrt python bindings to destroy stream
            cls._default_stream = None
        cls._active_stream = None


Stream.default_stream()
# Register the cleanup_default_stream method to be called at program exit
atexit.register(Stream.cleanup_default_stream)


@export.public_api(document_under="compiler")
class InputInfo:
    """
    Captures information about an input to a compiled function.
    """

    def __init__(
        self, shape: Sequence[Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]], dtype: "tripy.dtype"
    ) -> None:
        """
        Args:
            shape: The shape of the input.
                To indicate dynamic dimensions, provide the minimum, optimum, and maximum values for the dimension.
            dtype: The data type of the input.

        .. code-block:: python
            :linenos:
            :caption: Example

            inp = tp.InputInfo((2, 4), dtype=tp.float32)
            assert inp.shape_bounds.min == (2, 4)
            assert inp.shape_bounds.opt == (2, 4)
            assert inp.shape_bounds.max == (2, 4)

        .. code-block:: python
            :linenos:
            :caption: Dynamic Dimensions

            # The first dimension will support values in the range [1, 3], optimizing for a size of 2.
            inp = tp.InputInfo(((1, 2, 3), 4), dtype=tp.float32)
            assert inp.shape_bounds.min == (1, 4)
            assert inp.shape_bounds.opt == (2, 4)
            assert inp.shape_bounds.max == (3, 4)
        """
        # TODO (#252): Allow `shape` to be a shape tensor
        min_shape = []
        opt_shape = []
        max_shape = []
        for elem in shape:
            if isinstance(elem, numbers.Number):
                elem = (elem,) * 3
            elif isinstance(elem, Sequence):
                if not all(isinstance(val, numbers.Number) for val in elem):
                    raise_error(
                        "Shape values must be numbers.",
                        [f"Shape: {shape} contains an element: {repr(elem)} with non-numerical value(s)"],
                    )
                if len(elem) != 3:
                    raise_error(
                        "Incorrect number of shape values provided.",
                        [
                            f"Exactly 3 shape values must be provided for each dimension (min/opt/max)"
                            f" but got: {len(elem)} values in shape: {shape}. "
                        ],
                    )
            else:
                raise_error(
                    "Shape values should be either a single number or a Tuple specifying min/opt/max bounds.",
                    [f"Shape: {shape} contains an invalid element: {elem}"],
                )

            min_shape.append(elem[0])
            opt_shape.append(elem[1])
            max_shape.append(elem[2])

        self.shape_bounds = ShapeBounds(tuple(min_shape), tuple(opt_shape), tuple(max_shape))
        self.dtype = dtype

    def __str__(self) -> str:
        return f"InputInfo(min={self.shape_bounds.min}, opt={self.shape_bounds.opt}, max={self.shape_bounds.max}, dtype={self.dtype})"


# TODO(MLIR-TRT #923): Can generalize `InputInfo` and drop this class.
@export.public_api(document_under="compiler")
@dataclass
class ArgInfo:
    shape_bounds: Sequence[Tuple[int, int]]
    """A sequence of tuple(min, max) indicating the bounds of each dimension"""
    dtype: "tripy.dtype"
    """The datatype of the argument"""


@export.public_api(document_under="compiler")
class Executable:
    """
    Represents a compiled executable generated by the compiler.

    .. seealso:: :class:`Compiler`
    """

    # The constructor is intentionally undocumented because it is not meant to be called by users.
    # TODO(#155): output_devices is not needed after they can be queried from executable
    def __init__(self, executable, arg_names, output_devices):
        self._executable = executable
        self._executor = Executor(self._executable)
        self._arg_names = arg_names
        self._output_devices = output_devices
        self._executable_signature = self._executable.get_signature("main")

        # Build a signature so the executable works with `inspect.signature`
        params = []
        for name in self._arg_names:
            params.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Tensor))

        return_annotation = Tensor if self._executable_signature.get_num_output_args() == 1 else Sequence[Tensor]

        self.__signature__ = inspect.Signature(params, return_annotation=return_annotation)

    def __call__(self, *args, **kwargs) -> Union[Tensor, Sequence[Tensor]]:
        """
        Invokes the executable with the specified tensor arguments.

        Args:
            *args: Positional arguments. Must be of type :class:`Tensor` .
            **kwargs: Keyword arguments. Must be of type :class:`Tensor` .

        Returns:
            The output :class:`Tensor` s of the compiled function.


        .. code-block:: python
            :linenos:
            :caption: Example

            def add(a, b):
                return a + b

            # doc: no-print-locals compiler compiled_add
            compiler = tp.Compiler(add)
            compiled_add = compiler.compile(tp.InputInfo((1,), dtype=tp.float32), tp.InputInfo((1,), dtype=tp.float32))

            a = tp.ones((1,), dtype=tp.float32)
            b = tp.ones((1,), dtype=tp.float32)

            out = compiled_add(a, b)
        """
        input_tensors = []

        input_tensors.extend(args)
        # Need to get arguments in the order of self._arg_names, which may be different from kwargs ordering.
        expected_kwargs = self._arg_names[len(args) :]
        for name in expected_kwargs:
            if name not in kwargs:
                raise_error(f"Missing argument: {name}", [f"Expected the following arguments: {self._arg_names}"])

            input_tensors.extend(kwargs[name])
            del kwargs[name]

        if kwargs:
            raise_error(
                f"Extra keyword arguments: {list(kwargs.keys())}",
                [
                    f"Expected the following arguments: {self._arg_names}.\n"
                    f"Note: The following arguments were already provided as positional arguments: {self._arg_names[:len(args)]}"
                ],
            )

        # We do this after kwarg checks since those will be more informative (we can explain which arguments are missing/extra).
        num_args = len(args) + len(kwargs)

        if num_args != len(self._arg_names):
            raise_error(
                "Incorrect number of arguments.",
                [
                    f"Expected {len(self._arg_names)} arguments but got {num_args}.\n"
                    f"Note: Expected arguments were: {self._arg_names}",
                ],
            )

        # The executor expects concrete tensors as inputs, so we need to eval() here.
        for tensor in input_tensors:
            tensor.eval()

        try:
            executor_outputs = self._executor.execute(self._output_devices, input_tensors)
        except runtime.MTRTException as err:
            # TODO: Evaluate whether this should be moved into the executor
            if "function expects a memref type with element type" in str(err):
                # If the problem is a mismatched data type, we can provide a better error message than the executor can.
                expected_input_dtypes = [info.dtype for info in self.get_input_info()]
                for tensor, dtype, arg_name in zip(input_tensors, expected_input_dtypes, self._arg_names):
                    if tensor.dtype != dtype:
                        raise_error(
                            f"Unexpected tensor data type.",
                            [
                                f"For parameter {arg_name}, expected data type: {dtype} but got: {tensor.dtype}. Note: Argument was: ",
                                tensor,
                            ],
                        )
            elif "InternalError: failed to set input shape" in str(err) or "Runtime shape mismatch" in str(err):
                expected_input_shapes = [info.shape_bounds for info in self.get_input_info()]
                for tensor, expected_bounds, arg_name in zip(input_tensors, expected_input_shapes, self._arg_names):
                    shape = tensor.shape.data().data()
                    for i in range(len(shape)):
                        if shape[i] < expected_bounds[i][0] or shape[i] > expected_bounds[i][1]:
                            raise_error(
                                f"Unexpected tensor shape.",
                                [
                                    f"For parameter `{arg_name}`, expected the tensor shape `{tensor.shape}` to be within bounds for all dimensions. However, dimension {i} has a shape of {shape[i]}, which is not within the expected bounds of {expected_bounds[i]}. Note: The provided argument was: ",
                                    tensor,
                                ],
                            )
            raise

        # TODO (#192): avoid get_stack_info in runtime
        output_tensors = [Tensor(output) for output in executor_outputs]
        if len(output_tensors) == 1:
            output_tensors = output_tensors[0]
        return output_tensors

    def _get_arg_info(self, idx):
        arg = self._executable_signature.get_arg(idx)
        arg = runtime.MemRefType(arg)
        arg_bound = self._executable_signature.get_arg_bound(idx)
        shape_bounds = tuple(zip(arg_bound.min(), arg_bound.max()))
        if len(shape_bounds) == 0:
            # For static shape arguments, get_arg_bound returns an empty list and we fallback to arg.shape
            shape_bounds = tuple((x, x) for x in arg.shape)
        return ArgInfo(shape_bounds, mlir_utils.convert_runtime_dtype_to_tripy_dtype(arg.dtype))

    def get_input_info(self) -> Sequence[ArgInfo]:
        """
        Returns input tensors' information.

        Returns:
            A list containing one `ArgInfo` per input.

        .. code-block:: python
            :linenos:
            :caption: Get input info

            def add(a, b):
                return a + b

            # doc: no-print-locals compiler compiled_add
            compiler = tp.Compiler(add)
            compiled_add = compiler.compile(tp.InputInfo(([1, 2, 3],), dtype=tp.float32), tp.InputInfo(([1, 2, 3],), dtype=tp.float32))
            print(compiled_add.get_input_info())
        """
        input_info = []
        for idx in range(self._executable_signature.get_num_input_args()):
            input_info.append(self._get_arg_info(idx))
        return input_info

    def get_output_info(self) -> Sequence[ArgInfo]:
        """
        Returns output tensors' information.

        Returns:
            A list containing one `ArgInfo` per input.

        .. code-block:: python
            :linenos:
            :caption: Get output info

            def add(a, b):
                return a + b

            # doc: no-print-locals compiler compiled_add
            compiler = tp.Compiler(add)
            compiled_add = compiler.compile(tp.InputInfo(([1, 2, 3],), dtype=tp.float32), tp.InputInfo(([1, 2, 3],), dtype=tp.float32))
            print(compiled_add.get_output_info())
        """
        output_info = []
        offset = self._executable_signature.get_num_input_args()
        for idx in range(self._executable_signature.get_num_output_args()):
            output_info.append(self._get_arg_info(idx + offset))
        return output_info

    def save(self, path: str) -> None:
        """
        Saves the compiled executable to the given file.

        Args:
            path: The name of file to save the executable.

        .. code-block:: python
            :linenos:
            :caption: Save executable

            import os, tempfile

            def add(a, b):
                return a + b

            # doc: no-print-locals compiler compiled_add executable_file
            compiler = tp.Compiler(add)
            compiled_add = compiler.compile(tp.InputInfo(([1, 2, 3],), dtype=tp.float32), tp.InputInfo(([1, 2, 3],), dtype=tp.float32))

            with tempfile.TemporaryDirectory() as temp_dir:
                executable_file = os.path.join(temp_dir, "executable.json")
                compiled_add.save(executable_file)
                assert os.path.exists(executable_file)
        """
        json_utils.save(self, path)

    @classmethod
    def load(cls, path: str) -> "tripy.Executable":
        """
        Loads a compiled executable from a given directory.

        Args:
            path: The name of file to load the exectuable from.

        Returns:
            The executable object loaded from the file.

        .. code-block:: python
            :linenos:
            :caption: Save and load executable

            import os, tempfile

            def add(a, b):
                return a + b

            # doc: no-print-locals compiler compiled_add executable_file
            compiler = tp.Compiler(add)
            compiled_add = compiler.compile(tp.InputInfo(([1, 2, 3],), dtype=tp.float32), tp.InputInfo(([1, 2, 3],), dtype=tp.float32))

            with tempfile.TemporaryDirectory() as temp_dir:
                executable_file = os.path.join(temp_dir, "executable.json")
                compiled_add.save(executable_file)
                assert os.path.exists(executable_file)
                loaded_executable = tp.Executable.load(executable_file)
        """
        return json_utils.load(path)


@json_utils.Encoder.register(Executable)
def encode_executable(executable):
    return {
        "arg_names": executable._arg_names,
        "output_devices": executable._output_devices,
        "executable": base64.b64encode(executable._executable.serialize()).decode(),
    }


@json_utils.Decoder.register(Executable)
def decode_executable(executable_dict):
    executable_bytes = base64.b64decode(executable_dict["executable"])
    return Executable(
        runtime.Executable(executable_bytes),
        executable_dict["arg_names"],
        executable_dict["output_devices"],
    )


# TODO (#230): Support collections of tensors in args/kwargs
@export.public_api(document_under="compiler/index.rst")
class Compiler:
    """
    The compiler can compile Python functions into executables that run efficiently on the GPU.
    """

    def __init__(self, func: Callable, optimization_level: int = 3) -> None:
        """
        Args:
            func: The function or :class:`Module` to optimize. The function must satisfy the following requirements:

                - Must be a pure function with no side effects.
                  This means, for example, that you cannot use ``print`` or ``assert``.

                - Must not accept variadic positional or keyword arguments.

                - Must return one or more :class:`Tensor` s and no other types.

                The compiled function will have the following constraints:

                - Only :class:`Tensor` parameters to the function can become runtime inputs. All other types of parameters,
                  even collections of :class:`Tensor` s (e.g. ``List[Tensor]`` or ``Dict[str, Tensor]``), will be baked into
                  the compiled function as constants.

            optimization_level: The optimization level to use when compiling. Higher optimization levels can lead to better
                runtime performance at the cost of longer compile times.

        .. code-block:: python
            :linenos:
            :caption: Example

            def add(a, b):
                return a + b

            # doc: no-print-locals compiler compiled_add
            compiler = tp.Compiler(add)
            compiled_add = compiler.compile(tp.InputInfo((1,), dtype=tp.float32), tp.InputInfo((1,), dtype=tp.float32))

            a = tp.ones((1,), dtype=tp.float32)
            b = tp.ones((1,), dtype=tp.float32)

            out = compiled_add(a, b)
        """
        self.func = func
        self.optimization_level = optimization_level

        self._signature = inspect.signature(self.func)
        # TODO (#245): Support variadic arguments.
        if any(
            param.kind in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]
            for param in self._signature.parameters.values()
        ):
            raise_error("Variadic positional/keyword arguments are not currently supported.")

    def compile(self, *args, **kwargs) -> Executable:
        """
        Compiles the function. This works by first calling the function with the provided arguments
        in order to trace its execution, and the compiling the resulting traced graph.

        Parameters that should be runtime inputs in the compiled function should be provided
        as :class:`InputInfo` arguments to this function instead of as :class:`Tensor` s. Arguments
        of any other type will be treated as compile-time constants.

        Args:
            *args: Positional arguments to forward to the target function while tracing.
            **kwargs: Keyword arguments to forward to the target function while tracing.

        Returns:
            The compiled executable. This executable's parameters will be the subset of the original
            function's parameters for which :class:`InputInfo` s were provided to :func:`compile` and
            will only accept :class:`Tensor` arguments.


        .. code-block:: python
            :linenos:
            :caption: Dynamic Shapes

            def add(a, b):
                return a + b

            # doc: no-print-locals compiler compiled_add
            compiler = tp.Compiler(add)
            # Support shapes in the range of (1, 2) to (3, 2), optimizing for a shape of (2, 2)
            compiled_add = compiler.compile(tp.InputInfo(([1, 2, 3], 2), dtype=tp.float32), tp.InputInfo(([1, 2, 3], 2), dtype=tp.float32))

            small_a = tp.ones((1, 2), dtype=tp.float32)
            small_b = tp.ones((1, 2), dtype=tp.float32)

            small_out = compiled_add(small_a, small_b)

            # Now we can reuse the compiled function for any shapes within the range:
            big_a = tp.ones((3, 2), dtype=tp.float32)
            big_b = tp.ones((3, 2), dtype=tp.float32)

            big_out = compiled_add(big_a, big_b)


        .. code-block:: python
            :linenos:
            :caption: Baking Constants

            def add(a, b):
                return a + b

            # doc: no-print-locals compiler compiled_add
            compiler = tp.Compiler(add)

            # By using a non-InputInfo type (in this case, a Tensor) for the `b` argument to `compile`,
            # we are indicating that it is a compile-time constant. Consequently, the compiled function
            # will not accept `b` as an input.
            b = tp.ones((1,), dtype=tp.float32)
            compiled_add = compiler.compile(tp.InputInfo((1,), dtype=tp.float32), b)

            a = tp.ones((1,), dtype=tp.float32)

            # Note that we cannot provide `b` as an argument to the compiled function.
            out = compiled_add(a)
        """

        shapes = []
        trace_input_map = {}
        input_names = set()

        def process_arg(name, arg):
            if isinstance(arg, InputInfo):
                # Make new tensors for tracing.
                from tripy.common.datatype import floating, integer
                from tripy.frontend.trace.ops.fill import full

                init_value = 1 if issubclass(arg.dtype, integer) else 1.0 if issubclass(arg.dtype, floating) else True
                tensor = full(shape=arg.shape_bounds.opt, value=init_value, dtype=arg.dtype)
                tensor.name = name

                trace_input_map[name] = tensor
                shapes.append(arg.shape_bounds)
                input_names.add(name)

                return tensor
            return arg

        new_args = []
        for name, arg in utils.get_positional_arg_names(self.func, *args):
            new_args.append(process_arg(name, arg))

        new_kwargs = {}
        for name, arg in kwargs.items():
            new_kwargs[name] = process_arg(name, arg)

        # Figure out the signature of the compiled function. This should include only the arguments that were provided
        # as `InputInfo`s, but the order needs to match the signature of the original function.
        compiled_arg_names = [name for name in self._signature.parameters.keys() if name in input_names]

        trace_outputs = utils.make_list(self.func(*new_args, **new_kwargs))

        if not trace_outputs:
            raise_error(
                "Function must return 1 or more Tensors.",
                [f"Return value was: {repr(trace_outputs)}"],
            )

        for index, out in enumerate(trace_outputs):
            if not isinstance(out, Tensor):
                raise_error(
                    "Function must return 1 or more Tensors.",
                    [f"Return value {index} was not a tensor: {repr(out)}"],
                )

        # Order of trace inputs also needs to match that of the compiled_arg_names
        trace_inputs = [trace_input_map[name] for name in compiled_arg_names]
        trace = Trace(trace_outputs, trace_inputs, shapes=shapes)

        flat_ir = trace.to_flat_ir()
        mlir = flat_ir.to_mlir()
        compiler = MLIRCompiler(trt_builder_opt_level=self.optimization_level)
        executable = compiler.compile(mlir, flat_ir=flat_ir)

        return Executable(
            executable,
            compiled_arg_names,
            output_devices=[out.device for out in trace.outputs],
        )
