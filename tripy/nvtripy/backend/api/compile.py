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

import inspect
from typing import Any, Callable, Dict, Sequence

from nvtripy import export, utils
from nvtripy.backend.api.executable import Executable
from nvtripy.backend.api.input_info import InputInfo
from nvtripy.backend.mlir import Compiler as MLIRCompiler
from nvtripy.common.exception import raise_error
from nvtripy.frontend import Tensor, Trace


# TODO (#230): Support collections of tensors in args/kwargs
@export.public_api(document_under="compiling_code/compile.rst")
def compile(
    func: Callable, optimization_level: int = 3, *, args: Sequence[Any] = [], kwargs: Dict[str, Any] = {}
) -> Executable:
    """
    Compiles a function into an executable that runs efficiently on the GPU.

    This works by first calling the function with the provided arguments
    in order to trace its execution, and the compiling the resulting traced graph.

    Parameters that should be runtime inputs in the compiled function should be provided
    as :class:`InputInfo` arguments to this function instead of as :class:`Tensor` s. Arguments
    of any other type will be treated as compile-time constants.

    Args:
        func: The function or :class:`Module` to optimize. The function must satisfy the following requirements:

            - Must be a pure function with no side effects.
                This means, for example, that you cannot use ``print`` or ``assert``.

            - Must not accept variadic positional or keyword arguments.

            - Must return one or more :class:`Tensor` s and no other types.

            The compiled function will have the following constraints:

            - Only :class:`Tensor` parameters to the function can become runtime inputs.
                All other types of parameters, even collections of :class:`Tensor` s (e.g. ``List[Tensor]`` or ``Dict[str, Tensor]``),
                will be baked into the compiled function as constants.

        optimization_level: The optimization level to use when compiling. Higher optimization levels can lead to better
            runtime performance at the cost of longer compile times.

        args: Positional arguments to forward to the target function while tracing.
        kwargs: Keyword arguments to forward to the target function while tracing.

    Returns:
        The compiled executable. This executable's parameters will be the subset of the original
        function's parameters for which :class:`InputInfo` s were provided to :func:`compile` and
        will only accept :class:`Tensor` arguments.


    .. code-block:: python
        :linenos:
        :caption: Dynamic Shapes

        def add(a, b):
            return a + b

        # doc: no-print-locals compiled_add

        # Support shapes in the range of (1, 2) to (3, 2), optimizing for a
        # shape of (2, 2)
        compiled_add = tp.compile(
            add,
            args=[
                tp.InputInfo(shape=((1, 2, 3), 2), dtype=tp.float32),
                tp.InputInfo(shape=((1, 2, 3), 2), dtype=tp.float32),
            ],
        )

        small_a = tp.ones((1, 2), dtype=tp.float32)
        small_b = tp.ones((1, 2), dtype=tp.float32)

        small_out = compiled_add(small_a, small_b)

        # Now we can reuse the compiled function for any shapes within the
        # range:
        big_a = tp.ones((3, 2), dtype=tp.float32)
        big_b = tp.ones((3, 2), dtype=tp.float32)

        big_out = compiled_add(big_a, big_b)


    .. code-block:: python
        :linenos:
        :caption: Baking Constants

        def add(a, b):
            return a + b

        # doc: no-print-locals compiled_add

        # By using a non-InputInfo type (in this case, a Tensor) for the `b`
        # argument to `compile`, we are indicating that it is a compile-time
        # constant. Consequently, the compiled function will not accept `b`
        # as an input.
        b = tp.ones((1,), dtype=tp.float32)
        compiled_add = tp.compile(add, args=[tp.InputInfo((1,), dtype=tp.float32), b])

        a = tp.ones((1,), dtype=tp.float32)

        # Note that we cannot provide `b` as an argument to the compiled function.
        out = compiled_add(a)
    """
    signature = inspect.signature(func)
    # TODO (#245): Support variadic arguments.
    if any(
        param.kind in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]
        for param in signature.parameters.values()
    ):
        raise_error("Variadic positional/keyword arguments are not currently supported.")

    shapes = []
    trace_input_map = {}
    input_names = set()

    def process_arg(name, arg):
        if isinstance(arg, InputInfo):
            # Make new tensors for tracing.
            from nvtripy.common.datatype import floating, integer
            from nvtripy.frontend.trace.ops.fill import full

            init_value = 1 if issubclass(arg.dtype, integer) else 1.0 if issubclass(arg.dtype, floating) else True
            tensor = full(shape=arg.shape_bounds.opt, value=init_value, dtype=arg.dtype)
            tensor.name = name
            tensor.trace_tensor.is_compile_tracer = True

            trace_input_map[name] = tensor
            shapes.append(arg.shape_bounds)
            input_names.add(name)

            return tensor
        return arg

    new_args = []
    positional_arg_info, _ = utils.get_positional_arg_names(func, *args)
    for name, arg in positional_arg_info:
        new_args.append(process_arg(name, arg))

    new_kwargs = {}
    for name, arg in kwargs.items():
        new_kwargs[name] = process_arg(name, arg)

    # Figure out the signature of the compiled function. This should include only the arguments that were provided
    # as `InputInfo`s, but the order needs to match the signature of the original function.
    compiled_arg_names = [name for name in signature.parameters.keys() if name in input_names]

    trace_outputs = utils.make_list(func(*new_args, **new_kwargs))

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

    for op in trace.ops:
        for tensor in op.inputs + op.outputs:
            if tensor.is_compile_tracer and tensor.eval_stack_info is not None:
                raise_error(
                    "Cannot evaluate a tensor while compiling.", ["Tensor was evaluated here:", tensor.eval_stack_info]
                )

    flat_ir = trace.to_flat_ir()
    mlir = flat_ir.to_mlir()
    compiler = MLIRCompiler(trt_builder_opt_level=optimization_level)
    executable = compiler.compile(mlir, flat_ir=flat_ir)

    return Executable(
        executable,
        compiled_arg_names,
        output_devices=[out.device for out in trace.outputs],
    )
