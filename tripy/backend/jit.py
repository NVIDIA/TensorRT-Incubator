import atexit
import functools
from typing import Callable, Dict, Tuple

from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor
from tripy.common.logging import G_LOGGER
from tripy.flat_ir import FlatIR
from tripy.frontend import Tensor, nn


class jit:
    """
    Allows a function to be just-in-time compiled, which will replace the implementation of any pure
    function with a more efficient one.
    The implementation is cached such that all invocations after the first use the cached implementation
    instead of recompiling.
    """

    def __init__(self, func: Callable = None, **kwargs):
        """
        Args:
            func: Function to jit.

        Constraints:
            All Tensors are provided as args, not kwargs

        Using JIT as a decorator:
        ::

            import numpy as np

            a = tp.Tensor([1.0, 1.0], dtype=tp.float32, device=tp.device("gpu"))
            b = tp.Tensor([1.0, 1.0], dtype=tp.float32, device=tp.device("gpu"))

            @tp.jit
            def add(a, b):
                c = a + b
                return c

            out = add(a, b)

            assert (out.numpy() == np.array([2.0, 2.0])).all()

        Using JIT as a function:
        ::

            import numpy as np

            a = tp.Tensor([1.0, 1.0], dtype=tp.float32, device=tp.device("gpu"))
            b = tp.Tensor([1.0, 1.0], dtype=tp.float32, device=tp.device("gpu"))

            def add(a, b):
                c = a + b
                return c

            jit_add = tp.jit(add)
            out = jit_add(a, b)

            assert (out.numpy() == np.array([2.0, 2.0])).all()
        """
        self.kwargs = kwargs
        self.cache: Dict[Tuple, FlatIRExecutor] = {}
        self._const_args: Tuple[int] = ()
        self.func: Callable = None
        if func is not None:
            self.func = func
            self.decorated = self._helper(func)
        atexit.register(self.destroy)

    def _get_input_signature(self, origin_tensors, eval_tensors, kwargs) -> Tuple:
        """
        Returns the input signature, which is used as the cache key of a JIT function
        The function signature is a combination of:
          - Shapes of evaluated input arguments
          - IDs of constant arguments
          - Values of keyword arguments

        Args:
            origin_tensors: a list of original input Tensors
            eval_tensors: a list of evaluated input Tensors
            kwargs: keyword argument dict of the original function
        """
        input_args = []
        const_args = []
        for i, eval_arg in enumerate(eval_tensors):
            if eval_arg.const_fold:
                const_args.append(id(origin_tensors[i]))
            else:
                input_args.append(tuple(eval_arg.op.shape))

        input_sig = hash((*input_args, *const_args, tuple(sorted(kwargs.items()))))
        return input_sig

    def _helper(self, func: Callable):
        # Use self.kwargs here to check JIT arguments.
        # This method is what makes this class usable as a decorator.
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            # Eval triggers computation of input arguments which ensures that shape of inputs is known before
            # compiling and caching a function's implementation.
            eval_args = [
                Tensor(
                    arg.eval().view(),
                    dtype=arg.op.dtype,
                    device=arg.op.device,
                    shape=arg.op.shape,
                )
                for arg in args
            ]
            self._const_args = self.kwargs["const_argnums"] if "const_argnums" in self.kwargs else ()
            self._const_args = self._const_args + tuple(
                [index for index, arg in enumerate(args) if isinstance(arg, nn.Parameter)]
            )

            for i in range(len(eval_args)):
                if i not in self._const_args:
                    eval_args[i].const_fold = False
            eval_args = tuple(eval_args)

            # To know the shapes of input tensors, we need evaluated tensors
            cache_key = self._get_input_signature(args, eval_args, kwargs)
            if cache_key in self.cache:
                executor = self.cache[cache_key]
            else:
                return_tensors = func(*eval_args, **kwargs)
                if isinstance(return_tensors, Tensor):
                    return_tensors = [return_tensors]
                flat_ir = FlatIR(return_tensors)
                G_LOGGER.ir_printer(f"flatIR :\n{flat_ir}")
                output_devices = [o.device for o in flat_ir.outputs]

                compiler = FlatIRCompiler()
                executor = FlatIRExecutor(compiler.compile(flat_ir), output_devices)
                self.cache[cache_key] = executor

            outputs = executor.execute(eval_args)
            tensor_outputs = [
                Tensor(o.data.view(), device=out_device) for o, out_device in zip(outputs, executor.output_devices)
            ]
            if len(tensor_outputs) == 1:
                tensor_outputs = tensor_outputs[0]
            return tensor_outputs

        return decorated

    def __call__(self, *args, **kwargs):
        if callable(self.func):
            return self.decorated(*args, **kwargs)
        else:
            # args[0] represents the func passed as argument to jit. Ex: jitted_func = jit(func)
            return self._helper(args[0])

    def destroy(self):
        for _, executor in self.cache.items():
            executor.destroy()
