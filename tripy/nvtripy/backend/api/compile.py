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

import inspect
from typing import Any, Callable, Dict, Sequence

from nvtripy import constants, export, utils
from nvtripy.backend.api.executable import Executable
from nvtripy.backend.api.input_info import InputInfo
from nvtripy.backend.mlir import Compiler
from nvtripy.common.exception import raise_error
from nvtripy.frontend import Tensor, Trace
from nvtripy.frontend.module import Module
from nvtripy.utils.types import obj_name_or_type_name


# TODO (#230): Support collections of tensors in args/kwargs
@export.public_api(document_under="compiling_code/compile.rst")
def compile(
    func: Callable, optimization_level: int = 3, *, args: Sequence[Any] = [], kwargs: Dict[str, Any] = {}
) -> Executable:
    """
    Compiles a function into an executable that runs efficiently on the GPU.

    This works by first calling the function with the provided arguments
    in order to trace its execution, and the compiling the resulting traced graph.

    Parameters that should be runtime inputs to the compiled function should be provided
    as :class:`InputInfo` arguments to this function instead of as :class:`Tensor` s. Arguments
    of any other type will be treated as compile-time constants. Any constants should be in
    CPU memory.

    Args:
        func: The function or :class:`Module` to optimize. The function must satisfy the following requirements:

            - Must be a pure function with no side effects.
                This means, for example, that you cannot use ``print`` or ``assert``.

            - Must return one or more :class:`Tensor` s and no other types.

            The compiled function will have the following constraints:

            - Only :class:`Tensor` parameters to the function can become runtime inputs.
                All other types of parameters, even collections of :class:`Tensor` s
                (e.g. ``List[Tensor]`` or ``Dict[str, Tensor]``), will be baked into
                the compiled function as constants.

            - Variadic positional and keyword arguments will be "frozen" by this function.
                That is, only the arguments supplied to ``args`` and ``kwargs`` when compiling
                will be available at runtime.

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

        small_a = tp.ones((1, 2), dtype=tp.float32).eval()
        small_b = tp.ones((1, 2), dtype=tp.float32).eval()

        small_out = compiled_add(small_a, small_b)

        # Now we can reuse the compiled function for any shapes within the
        # range:
        big_a = tp.ones((3, 2), dtype=tp.float32).eval()
        big_b = tp.ones((3, 2), dtype=tp.float32).eval()

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

        a = tp.ones((1,), dtype=tp.float32).eval()

        # Note that we cannot provide `b` as an argument to the compiled function.
        out = compiled_add(a)
    """
    signature = inspect.signature(func)

    trace_input_map = {}
    input_names = set()
    input_infos = {}

    # Set up names for the weights in the module to make the trace easier to read.
    if isinstance(func, Module):
        for name, weight in func.state_dict().items():
            weight.name = name

    def process_arg(name, arg):
        if isinstance(arg, InputInfo):
            # Make new tensors for tracing.
            from nvtripy.common.datatype import floating, integer
            from nvtripy.frontend.ops.full import full

            input_infos[name] = arg

            init_value = 1 if issubclass(arg.dtype, integer) else 1.0 if issubclass(arg.dtype, floating) else True
            tensor = full(shape=arg.shape_bounds.opt, value=init_value, dtype=arg.dtype)
            tensor.name = name
            tensor.trace_tensor.is_compile_tracer = True

            # Set trace tensor dimensions for any static dimensions
            tensor.trace_tensor.shape = tuple(
                constants.DYNAMIC_DIM if dim_min != dim_max else dim_min
                for dim_min, dim_max in zip(arg.shape_bounds.min, arg.shape_bounds.max)
            )

            trace_input_map[name] = tensor
            input_names.add(name)

            return tensor
        return arg

    compiled_arg_names = []

    new_args = []
    positional_arg_info, variadic_info = utils.utils.get_positional_arg_names(func, *args)

    varargs_name = None
    varargs_index = None
    if variadic_info is not None:
        varargs_name, varargs_index = variadic_info

    for index, (name, arg) in enumerate(positional_arg_info):
        # For variadic arguments, update the name with an index to make it unique
        if name == varargs_name:
            name += str(index - varargs_index)

        new_args.append(process_arg(name, arg))
        compiled_arg_names.append(name)

    new_kwargs = {}
    ordered_kwargs = []
    # First add kwargs that are in the signature in order
    for param in signature.parameters.values():
        if param.name in kwargs:
            ordered_kwargs.append(param.name)
            new_kwargs[param.name] = process_arg(param.name, kwargs[param.name])

    # Then add any remaining kwargs (i.e. variadic ones) in their original order
    for name in kwargs:
        if name not in ordered_kwargs:
            ordered_kwargs.append(name)
            new_kwargs[name] = process_arg(name, kwargs[name])

    compiled_arg_names.extend(ordered_kwargs)

    # Figure out the signature of the compiled function. This should include only the arguments that were provided
    # as `InputInfo`s, but the order needs to match the signature of the original function.
    compiled_arg_names = [name for name in compiled_arg_names if name in input_names]

    func_out = func(*new_args, **new_kwargs)
    trace_outputs = utils.utils.make_list(func_out)

    if not trace_outputs:
        raise_error(
            "Function must return 1 or more Tensors.",
            [f"Return value was: {repr(trace_outputs)}"],
        )

    for index, trace_out in enumerate(trace_outputs):
        if not isinstance(trace_out, Tensor):
            raise_error(
                "Function must return 1 or more Tensors.",
                [f"Return value {index} was not a tensor: {repr(trace_out)}"],
            )

    # Order of trace inputs also needs to match that of the compiled_arg_names
    trace_inputs = [trace_input_map[name].trace_tensor for name in compiled_arg_names]
    trace = Trace(
        [tensor.trace_tensor for tensor in trace_outputs],
        trace_inputs,
        input_infos=input_infos,
        name=obj_name_or_type_name(func),
    )

    for op in trace.ops:
        for tensor in op.inputs + op.outputs:
            if tensor.is_compile_tracer and tensor.eval_stack_info is not None:
                raise_error(
                    "Cannot evaluate a tensor while compiling.", ["Tensor was evaluated here:", tensor.eval_stack_info]
                )

    mlir = trace.to_mlir()
    compiler = Compiler(trt_builder_opt_level=optimization_level)
    executable = compiler.compile(mlir, trace=trace)

    assert isinstance(func_out, Tensor) or isinstance(
        func_out, Sequence
    ), "This function is only implemented for Tensors or sequences of Tensors"
    return Executable(
        executable,
        compiled_arg_names,
        return_single_tensor_as_sequence=isinstance(func_out, Sequence),
        input_infos=input_infos,
    )
