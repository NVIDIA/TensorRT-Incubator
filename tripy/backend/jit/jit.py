import ast
import functools
import inspect
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

from tripy import export, utils
from tripy.backend.jit.cached_executable import CachedExecutable
from tripy.backend.jit.dynamic_storage import DynamicStorage
from tripy.backend.jit.utils import get_trace_signature
from tripy.backend.mlir.compiler import Compiler
from tripy.backend.mlir.executor import Executor
from tripy.backend.utils import TensorInfo, get_tensor_info, get_devices
from tripy.common.device import device
from tripy.common.exception import raise_error
from tripy.frontend import Shape, Tensor
from tripy.frontend.module import Module, Parameter
from tripy.frontend.trace import Trace
from tripy.frontend.trace.ops import Storage
from tripy.logging import logger


@export.public_api(autodoc_options=[":no-special-members:"])
class jit:
    """
    Indicates that a function should be just-in-time compiled to an executable the first time it is used.
    The function must be pure and will be replaced by a more efficient implementation.

    Executables are cached such that subsequent invocations do not need to recompile
    unless the arguments provided are incompatible with the previously compiled executable(s).
    """

    def __init__(self, func: Callable = None, const_argnums: Tuple = (), optimization_level: int = 3) -> None:
        """
        Args:
            func: A pure function.
            const_argnums: Argument indices of func that should be treated as compile-time constants.
            optimization_level: An integer argument that specifies optimization level allowed for the underlying compiler.
                                Level 0 enables the fastest compilation and level 5 enables the slowest compilation.
                                Valid range: [0, 5]. Defaults to 3.

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

            assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 2.0]))

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

            assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 2.0]))
        """
        self.cache: Dict[str, List[CachedExecutable]] = defaultdict(list)
        self.const_argnums = const_argnums
        self.optimization_level = optimization_level
        if not isinstance(self.optimization_level, int) or self.optimization_level < 0 or self.optimization_level > 5:
            raise ValueError(
                f"Optimization level must be an integer within the range [0, 5], got {self.optimization_level}"
            )
        # Caching is currently a bit complicated - the key of `self.cache` relies
        # on determining the signature of the trace. However, within an instance of
        # a JIT object, the trace signature can only change if the **kwargs
        # (and a few other parameters - look for HACK (#109)) change.
        # This means that we can avoid recomputing the cache key (and hence tracing)
        # by caching the keys for a given set of **kwargs. Effectively, we have a cache
        # for the cache keys.
        self._trace_signatures: Dict[int, int] = {}

        # Note that since tp.Shape (frontend shapes) are handled only at the front-end,
        # we also need to cache which outputs to treat as shapes or else we would
        # have no way to reconstruct it
        self._frontend_shape_outputs: Dict[int, Tuple[int]] = {}

        self._func: Callable = None
        self._decorated: Callable = None
        self._obj = None

        if func is not None:
            self._func = func
            self._decorated = self._helper(func)

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
            const_argnums = self.const_argnums + tuple(
                index for index, arg in enumerate(args) if isinstance(arg, Parameter)
            )

            # HACK (#109): If the constant input tensors change, we need to recompile - that means we need
            #   to retrigger tracing and then also modify the trace signature.
            const_tensor_ids = tuple(id(args[idx]) for idx in const_argnums)

            # since we are caching which outputs are frontend shapes and that can
            # be affected by which *inputs* are frontend shapes, we should include this
            # in the cache key
            shape_input_info = tuple(isinstance(arg, Shape) for arg in args)

            inputs = []
            input_tensor_info = []
            # For the purposes of tracing, constant arguments are not considered inputs.
            trace_inputs = []

            for index, arg in enumerate(args):
                # Creating a new tensor to make sure each arg has a unique name
                # Copy stack information from the original tensor so error messages are better.
                dynamic_shape = arg._dynamic_shape
                tensor = Tensor(arg.eval(), dynamic_shape, stack_info=arg.stack_info)
                if isinstance(arg, Shape):
                    tensor = Shape(tensor)
                # Replace the tensor op with one that can supply dynamic shapes.
                storage = tensor.trace_tensor.producer
                assert isinstance(storage, Storage), f"{tensor} should have had a storage op after `eval()`"
                DynamicStorage.build_internal([], [tensor.trace_tensor], storage.data, dynamic_shape)

                if index not in const_argnums:
                    trace_inputs.append(tensor)
                inputs.append(tensor)
                input_tensor_info.append(
                    TensorInfo(
                        len(tensor.trace_tensor.producer.shape),
                        tensor.trace_tensor.producer.shape,
                        tensor.trace_tensor.producer.dtype,
                        tensor.trace_tensor.producer.device,
                    )
                )

            def warn_if_user_code_has_illegal_code(illegal_condition: Callable, illegal_warning_prefix: str):
                # Map from method name to statement where node is found.
                node_statements: Dict[str, List[str]] = defaultdict(list)

                # Raise warning if using print in jit
                if inspect.isfunction(func):
                    node_statements[func.__class__.__name__].extend(utils.find_node_in_method(func, illegal_condition))
                else:

                    def walk_through_class(user_class):
                        if isinstance(user_class, Module):
                            node_statements[user_class.__class__.__name__].extend(
                                utils.find_node_in_method(user_class.__call__, illegal_condition)
                            )

                            for child_name, child in user_class.named_children():
                                walk_through_class(child)

                    walk_through_class(func)

                # Create warning message for the user.
                warning_messages = []
                for func_name, statements in node_statements.items():
                    if statements:
                        formatted_statements = ", ".join(statements)
                        warning_messages.append(f"'{func_name}' : {formatted_statements}")

                combined_warning_message = "\n".join(warning_messages)
                if combined_warning_message:
                    logger.warning(illegal_warning_prefix + combined_warning_message)

            # The first time we run the function, we compute the cache key by looking at the signature of the
            # trace. On subsequent invocations, we skip this step.
            # Note that we also have to take note of and preserve which inputs should be treated as tp.Shape,
            # since this information is tracked only at the front-end.
            def make_trace():
                try:

                    illegal_checks: List[tuple[Callable, str]] = [
                        # TODO (#107): Revisit the warning when tripy.print is implemented for suggested usage.
                        (
                            lambda node, _: isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print",
                            "Usage of print statement in jitted functions is not recommended, instead use tripy.print to print in all invocations of the function.\n",
                        ),
                        # Check if dynamic shape tensor is initialized in jitted function.
                        (
                            lambda node, source: isinstance(node, ast.Assign)
                            and all(substr in source[node.lineno - 1].strip() for substr in ["Tensor", "shape="]),
                            "Initializing dynamic shape tensor in jitted functions is not recommended and Tripy will ignore the shape range.",
                        ),
                        # Check if pdb is used in jitted function
                        (
                            lambda node, _: isinstance(node, ast.Call)
                            and hasattr(node.func, "attr")
                            and node.func.attr in {"set_trace"}
                            and getattr(node.func.value, "id", "") == "pdb",
                            "Using pdb inside jitted function is not recommended as this will break the jitted function scope.",
                        ),
                    ]

                    for check_callable, warning_message in illegal_checks:
                        warn_if_user_code_has_illegal_code(check_callable, warning_message)
                except:
                    pass

                if self._obj is not None:
                    outputs = func(self._obj, *inputs, **kwargs)
                else:
                    outputs = func(*inputs, **kwargs)
                if isinstance(outputs, Tensor):
                    outputs = [outputs]

                # we need to pass shape information in order to preserve it
                outputs_as_shapes = [isinstance(output, Shape) for output in outputs]
                return Trace(outputs, trace_inputs), outputs_as_shapes

            # See _trace_signatures in __init__ for an explanation of this.
            # Note that the `trace_signature_key` is local to this instance so we can safely use Python's
            # built-in `hash()` even though it's randomized.
            #
            # We also need to include the pattern of which arguments are front-end shapes because we cache
            # which *outputs* are shapes and the inputs affect that.
            trace_signature_key = hash(tuple(kwargs.items()) + const_tensor_ids + shape_input_info)

            trace = None

            output_devices = None

            if trace_signature_key not in self._trace_signatures:

                def compute_trace_signature(trace):
                    from tripy import __version__

                    # HACK (#109): Treat different constant inputs as different traces.
                    # Similarly, treat different patterns of tp.Shape inputs likewise
                    return str(utils.md5(__version__, get_trace_signature(trace), const_tensor_ids, shape_input_info))

                trace, outputs_as_shape = make_trace()
                self._trace_signatures[trace_signature_key] = compute_trace_signature(trace)
                self._frontend_shape_outputs[trace_signature_key] = tuple(outputs_as_shape)
                input_tensor_info = get_tensor_info(trace.inputs)
                output_tensor_info = get_tensor_info(trace.outputs)
                output_devices = get_devices(output_tensor_info)

            tensors_on_host = [x for x in args if x.trace_tensor.producer.device.kind != "gpu"]
            if len(tensors_on_host) > 0:
                raise_error(
                    f"JIT requires all the inputs to be on GPU.",
                    details=["The following input tensors are not on the GPU:"] + tensors_on_host,
                )

            trace_signature = self._trace_signatures[trace_signature_key]
            outputs_as_shape = self._frontend_shape_outputs[trace_signature_key]
            for executable, executor in self.cache.get(trace_signature, []):
                if executable.is_compatible(input_tensor_info):
                    break
            else:
                if trace is None:
                    trace, outputs_as_shape = make_trace()
                flat_ir = trace.to_flat_ir()
                mlir = flat_ir.to_mlir()
                compiler = Compiler(trt_builder_opt_level=self.optimization_level)
                executable = CachedExecutable(
                    compiler.compile(mlir, flat_ir=flat_ir),
                    get_tensor_info(trace.inputs),
                    get_devices(get_tensor_info(trace.outputs)),
                )
                executor = Executor(
                    executable.executable,
                )
                self.cache[trace_signature].append((executable, executor))

            # HACK (#155): We only use the executable information if we didn't recompute the trace.
            # filter out const-folded inputs
            outputs = executor.execute(
                utils.default(output_devices, executable.output_devices),
                trace_inputs,
            )

            # TODO(#192): avoid get_stack_info in runtime
            # preserve which outputs are tp.Shape
            tensor_outputs = [
                Shape(output) if outputs_as_shape[i] else Tensor(output) for i, output in enumerate(outputs)
            ]

            if len(tensor_outputs) == 1:
                tensor_outputs = tensor_outputs[0]
            return tensor_outputs

        return decorated

    def __call__(self, *args, **kwargs):
        def check_if_args_part_of_jit_func(args):
            # This function checks if any inputs traced back from args lead to DynamicStorage op.
            # Note that this a proxy for verifying if args are in jit function or not.
            # It not possible to verify from just the frontend tensor since the frontend tensor can be
            # part of multiple scopes.
            found_dynamic_storage = False
            exprs = [tensor.trace_tensor.producer for tensor in args]
            seen_op_ids = set()
            while exprs:
                head = exprs.pop(0)
                if isinstance(head, DynamicStorage):
                    found_dynamic_storage = True

                if id(head) in seen_op_ids:
                    continue

                exprs.extend([inp.producer for inp in head.inputs])

            return found_dynamic_storage

        if callable(self._func):
            if check_if_args_part_of_jit_func(args):
                if self._obj is not None:
                    return self._func(self._obj, *args, **kwargs)
                else:
                    return self._func(*args, **kwargs)
            return self._decorated(*args, **kwargs)
        else:
            # jit decorator with kwargs: @jit(...) triggers both __init__ and __call__
            self._func = args[0]
            self._decorated = self._helper(self._func)
            return self
