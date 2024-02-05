import atexit
import ast
import functools
import inspect
import textwrap
from collections import defaultdict
from typing import Callable, Dict, List

from tripy import utils
from tripy.backend.jit.cached_executable import CachedExecutable
from tripy.backend.jit.utils import TensorInfo, get_trace_signature, get_tensor_info
from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor
from tripy.common.logging import G_LOGGER
from tripy.frontend import Tensor, nn
from tripy.frontend.trace import Trace
from tripy.frontend.nn.module import Module


class jit:
    """
    Indicates that a function should be just-in-time compiled to an executable the first time it is used.
    The function must be pure and will be replaced by a more efficient implementation.

    Executables are cached such that subsequent invocations do not need to recompile
    unless the arguments provided are incompatible with the previously compiled executable(s).
    """

    def __init__(self, func: Callable = None, **kwargs):
        """
        Args:
            func: A pure function.

        Constraints:
            All :class:`tripy.Tensor` arguments must be provided as positional arguments and not keyword arguments.

        .. code-block:: python
            :linenos:
            :caption: JIT As A Decorator

            a = tp.Tensor([1.0, 1.0], dtype=tp.float32, device=tp.device("gpu"))
            b = tp.Tensor([1.0, 1.0], dtype=tp.float32, device=tp.device("gpu"))

            @tp.jit
            def add(a, b):
                c = a + b
                return c

            output = add(a, b)

            assert np.array_equal(output.numpy(), np.array([2.0, 2.0]))

        .. code-block:: python
            :linenos:
            :caption: JIT As A Function

            a = tp.Tensor([1.0, 1.0], dtype=tp.float32, device=tp.device("gpu"))
            b = tp.Tensor([1.0, 1.0], dtype=tp.float32, device=tp.device("gpu"))

            def add(a, b):
                c = a + b
                return c

            jit_add = tp.jit(add)
            output = jit_add(a, b)

            assert np.array_equal(output.numpy(), np.array([2.0, 2.0]))
        """
        self.kwargs = kwargs
        self.cache: Dict[str, List[CachedExecutable]] = defaultdict(list)

        # Caching is currently a bit complicated - the key of `self.cache` relies
        # on determining the signature of the trace. However, within an instance of
        # a JIT object, the trace signature can only change if the **kwargs
        # (and a few other parameters - look for HACK (#109)) change.
        # This means that we can avoid recomputing the cache key (and hence tracing)
        # by caching the keys for a given set of **kwargs. Effectively, we have a cache
        # for the cache keys.
        self._trace_signatures: Dict[int, int] = {}

        self._func: Callable = None
        self._decorated: Callable = None
        self._obj = None

        if func is not None:
            self._func = func
            self._decorated = self._helper(func)
        atexit.register(self.destroy)

    def __get__(self, obj, type=None):
        # This function is required to make the decorator work correctly with methods.
        self._obj = obj
        return self

    def _helper(self, func: Callable):
        # Use self.kwargs here to check JIT arguments.
        # This method is what makes this class usable as a decorator.
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            # Eval triggers computation of input arguments which ensures that shape of inputs is known before
            # compiling and caching a function's implementation.
            const_argnums = self.kwargs.get("const_argnums", tuple()) + tuple(
                index for index, arg in enumerate(args) if isinstance(arg, nn.Parameter)
            )

            # HACK (#109): If the constant input tensors change, we need to recompile - that means we need
            #   to retrigger tracing and then also modify the trace signature.
            const_tensor_ids = tuple(id(args[idx]) for idx in const_argnums)

            inputs = []
            input_tensor_info = []
            # For the purposes of tracing, constant arguments are not considered inputs.
            trace_inputs = []

            for index, arg in enumerate(args):
                # Creating a new tensor to make sure each arg has a unique name
                tensor = Tensor(arg.eval())
                tensor._stack_info = arg._stack_info
                if index not in const_argnums:
                    trace_inputs.append(tensor)
                inputs.append(tensor)
                input_tensor_info.append(TensorInfo(tensor.op.shape, tensor.op.dtype, tensor.op.device))

            def warn_if_user_code_has_print():
                # Map from method name to print statement.
                print_statements: Dict[str, List[str]] = defaultdict(list)

                # Raise warning if using print in jit
                if inspect.isfunction(func):
                    print_statements[func.__class__.__name__].extend(utils.find_node_in_method(func, "print"))
                else:

                    def walk_through_class(user_class):
                        if isinstance(user_class, Module):
                            print_statements[user_class.__class__.__name__].extend(
                                utils.find_node_in_method(user_class.__call__, "print")
                            )

                            for child_name, child in user_class.named_children():
                                walk_through_class(child)

                    walk_through_class(func)

                # Create warning message for the user.
                warning_messages = []
                for func_name, statements in print_statements.items():
                    if statements:
                        formatted_statements = ", ".join(statements)
                        warning_messages.append(f"'{func_name}' uses 'print' statements: {formatted_statements}")

                combined_warning_message = "\n".join(warning_messages)
                if combined_warning_message:
                    # TODO (#107): Revisit the warning when tripy.print is implemented for suggested usage.
                    G_LOGGER.warning(
                        f"Usage of print statement in jitted functions is not recommended, instead use tripy.print to print in all invocations of the function.\n"
                        + combined_warning_message
                    )

            # The first time we run the function, we compute the cache key by looking at the signature of the
            # trace. On subsequent invocations, we skip this step.
            def make_trace():
                try:
                    warn_if_user_code_has_print()
                except:
                    pass

                if self._obj is not None:
                    outputs = func(self._obj, *inputs, **kwargs)
                else:
                    outputs = func(*inputs, **kwargs)
                if isinstance(outputs, Tensor):
                    outputs = [outputs]
                return Trace(outputs, trace_inputs)

            # See _trace_signatures in __init__ for an explanation of this.
            # Note that the `trace_signature_key` is local to this instance so we can safely use Python's
            # built-in `hash()` even though it's randomized.
            #
            # HACK (#109): If the input shapes change, we need to retrace so that we can determine the new output
            #   shapes. This is only required because our executor currently needs the output memory up front.
            dim_list = []
            for inp in inputs:
                dim_list.extend([dim.runtime_value for dim in inp.op.shape])

            trace_signature_key = hash(tuple(kwargs.items()) + const_tensor_ids + tuple(dim_list))

            trace = None
            output_tensor_info = None

            if trace_signature_key not in self._trace_signatures:

                def compute_trace_signature(trace):
                    from tripy import __version__

                    # HACK (#109): Treat different constant inputs as different traces.
                    return str(utils.md5(__version__, get_trace_signature(trace), const_tensor_ids))

                trace = make_trace()
                G_LOGGER.ir_printer(f"Trace :\n{trace}")
                self._trace_signatures[trace_signature_key] = compute_trace_signature(trace)
                input_tensor_info = get_tensor_info(trace.inputs)
                output_tensor_info = get_tensor_info(trace.outputs)

            trace_signature = self._trace_signatures[trace_signature_key]
            for executable in self.cache.get(trace_signature, []):
                if executable.is_compatible(input_tensor_info):
                    break
            else:
                if trace is None:
                    trace = make_trace()
                flat_ir = trace.to_flat_ir()

                compiler = FlatIRCompiler()
                executable = CachedExecutable(
                    compiler.compile(flat_ir),
                    get_tensor_info(trace.inputs),
                    get_tensor_info(trace.outputs),
                )
                self.cache[trace_signature].append(executable)

            executor = FlatIRExecutor(
                executable.executable,
                # HACK (#109): We only use the executables I/O tensor information if we didn't recompute the trace.
                utils.default(input_tensor_info, executable.input_info),
                utils.default(output_tensor_info, executable.output_info),
            )
            # filter out const-folded inputs
            outputs = executor.execute(trace_inputs)

            tensor_outputs = [Tensor(output) for output in outputs]
            if len(tensor_outputs) == 1:
                tensor_outputs = tensor_outputs[0]
            return tensor_outputs

        return decorated

    def __call__(self, *args, **kwargs):
        if callable(self._func):
            return self._decorated(*args, **kwargs)
        else:
            # jit decorator with kwargs: @jit(...) triggers both __init__ and __call__
            self._func = args[0]
            self._decorated = self._helper(self._func)
            return self

    def destroy(self):
        from tripy.backend.mlir.mlir import mlir_wrapper

        mlir_backend = mlir_wrapper()
        for _, cached_executables in self.cache.items():
            for cached_executable in cached_executables:
                mlir_backend.exec_destroy(cached_executable.executable)
