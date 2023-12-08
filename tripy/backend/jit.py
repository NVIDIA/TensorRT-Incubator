import atexit
import functools
from typing import Callable, Dict, List, Tuple

from tripy.common.device import device
from tripy.util.util import make_list
from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor
from tripy.flat_ir import FlatIR
from tripy.frontend.tensor import Tensor
from tripy.ops import Storage
from tripy.common.logging import G_LOGGER


class JIT:
    """
    Represents JIT compilation which can be used to get an efficient implementation of any pure function.
    The implementation is cached such that all invocations after the first use the cached implementation.
    This interface provides capability to use jit as decorator and as a function.
    """

    def __init__(self, func: Callable = None, **kwargs):
        """
        Args:
            func: Function to jit.

        Constraints:
            All Tensors are provided as args, not kwargs

        Example:
        ::
            import tripy
            import numpy as np
            from tripy.common.logging import set_logger_mode, LoggerModes
            set_logger_mode(LoggerModes.IR | LoggerModes.TIMING | LoggerModes.VERBOSE)

            # JIT as a decorator example.
            @tripy.jit
            def adder(a, b):
                c = a + b
                return c

            a = tripy.Tensor(np.ones(1, dtype=np.float32), device=tripy.device("gpu"))
            b = tripy.Tensor(np.ones(1, dtype=np.float32), device=tripy.device("gpu"))

            out_decorator = adder(a, b)

            # JIT as a function example.
            def adder(a, b):
                c = a + b
                return c

            jitted_func = tripy.jit(adder)
            out_func = jitted_func(a, b)
            assert out_decorator.eval() == out_func.eval()
        """
        self.kwargs = kwargs
        self.cache: Dict[Tuple, FlatIRExecutor] = {}
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
            # TODO: make arg.eval() return Storage on the same device
            eval_args = [
                Tensor(list(arg.eval()), dtype=arg.op.dtype, device=device("gpu"), shape=arg.op.shape) for arg in args
            ]
            const_argnums = self.kwargs["const_argnums"] if "const_argnums" in self.kwargs else ()
            for i in range(len(eval_args)):
                if i not in const_argnums:
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

                compiler = FlatIRCompiler()
                executor = FlatIRExecutor(compiler.compile(flat_ir))
                self.cache[cache_key] = executor

            outputs = executor.execute(eval_args)
            tensor_outputs = [Tensor(o) for o in outputs]
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
