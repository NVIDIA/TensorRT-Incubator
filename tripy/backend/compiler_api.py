import inspect
import numbers
from textwrap import dedent
from typing import Callable, Sequence, Tuple, Union

import mlir_tensorrt.runtime.api as runtime

from tripy import export, utils
from tripy.backend.mlir import Compiler as MLIRCompiler
from tripy.backend.mlir import Executor
from tripy.common.exception import raise_error
from tripy.common.shape_bounds import ShapeBounds
from tripy.frontend import Tensor, Trace


@export.public_api(document_under="compiler")
class InputInfo:
    """
    Represents an input to a compiled function.
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


# TODO (#230): Support collections of tensors in args/kwargs
@export.public_api(document_under="compiler")
class Compiler:
    """
    The Tripy compiler.
    """

    def __init__(self, func: Callable, optimization_level: int = 3) -> None:
        """
        Args:
            func: The function or :class:`Module` to optimize. The function must satisfy the following requirements:

                - Must be a pure function with no side effects.
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

            # doc: no-print-locals compiler
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

    def compile(self, *args, **kwargs) -> Callable:
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
            The compiled function. This function's parameters will be the subset of the original
            function's parameters for which :class:`InputInfo` s were provided to :func:`compile` and
            will only accept :class:`Tensor` arguments.

        .. code-block:: python
            :linenos:
            :caption: Dynamic Shapes

            def add(a, b):
                return a + b

            # doc: no-print-locals compiler
            compiler = tp.Compiler(add)
            # Support shapes in the range of (1,) to (3,), optimizing for a shape of (2,)
            compiled_add = compiler.compile(tp.InputInfo(([1, 2, 3],), dtype=tp.float32), tp.InputInfo(([1, 2, 3],), dtype=tp.float32))

            small_a = tp.ones((1,), dtype=tp.float32)
            small_b = tp.ones((1,), dtype=tp.float32)

            small_out = compiled_add(small_a, small_b)

            # Now we can reuse the compiled function for any shapes within the range:
            big_a = tp.ones((3,), dtype=tp.float32)
            big_b = tp.ones((3,), dtype=tp.float32)

            big_out = compiled_add(big_a, big_b)


        .. code-block:: python
            :linenos:
            :caption: Baking Constants

            def add(a, b):
                return a + b

            # doc: no-print-locals compiler
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
                from tripy.frontend.ops import ones

                tensor = ones(shape=arg.shape_bounds.opt, dtype=arg.dtype)
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
        executor = Executor(executable)

        expected_input_dtypes = [inp.dtype for inp in trace_inputs]

        def execute(input_tensors, arg_names):
            # The executor expects concrete tensors as inputs, so we need to eval() here.
            for tensor in input_tensors:
                tensor.eval()

            try:
                executor_outputs = executor.execute([out.device for out in trace.outputs], input_tensors)
            except runtime.MTRTException as err:
                # TODO: Evaluate whether this should be moved into the executor
                if "function expects a memref type with element type" in str(err):
                    # If the problem is a mismatched data type, we can provide a better error message than the executor can.
                    for tensor, dtype, arg_name in zip(input_tensors, expected_input_dtypes, arg_names):
                        if tensor.dtype != dtype:
                            raise_error(
                                f"Unexpected tensor data type.",
                                [
                                    f"For parameter {arg_name}, expected data type: {dtype} but got: {tensor.dtype}. Note: Argument was: ",
                                    tensor,
                                ],
                            )
                raise

            # TODO (#192): avoid get_stack_info in runtime
            output_tensors = [Tensor(output) for output in executor_outputs]
            if len(output_tensors) == 1:
                output_tensors = output_tensors[0]
            return output_tensors

        new_locals = {}
        compiled_arg_names_str = ", ".join(compiled_arg_names)
        exec(
            dedent(
                f"""
                def compiled_func({compiled_arg_names_str}):
                    input_tensors = [{compiled_arg_names_str}]
                    return execute(input_tensors, {list(map(repr, compiled_arg_names))})
                """
            ),
            {"execute": execute},
            new_locals,
        )

        # TODO (#246): Return something that can be queried programatically to determine arguments/return values.
        return new_locals["compiled_func"]
