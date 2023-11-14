import functools
from typing import Callable, Dict, Tuple, List

from tripy.logging import G_LOGGER
from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor
from tripy.flat_ir import FlatIR
from tripy.frontend.tensor import Tensor


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

        Example:
        ::
            from tripy.frontend import Tensor
            from tripy.jit import JIT as jit
            import numpy as np

            # JIT as a decorator example.
            @jit
            def adder(a, b):
                c = a + b
                return c

            a = Tensor(np.ones(1, dtype=np.float32))
            b = Tensor(np.ones(1, dtype=np.float32))

            out_decorator = adder(a, b)

            # JIT as a function example.
            def adder(a, b):
                c = a + b
                return c

            jitted_func = jit(adder)
            out_func = jitted_func(a, b)
            assert out_decorator.eval() == out_func.eval()
        """
        self.kwargs = kwargs
        self.cache: Dict[Tuple, List] = {}  # Todo: Cache key should be inputs (named dim with their dimension range)
        self.func: Callable = None
        if func is not None:
            self.func = func
            self.decorated = self._helper(func)

    def _helper(self, func: Callable):
        # Use self.kwargs here to check JIT arguments.
        # This method is what makes this class usable as a decorator.
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            cache_key = (args, tuple(sorted(kwargs.items())))
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Eval triggers computation of input arguments which ensures that shape of inputs is known before
            # compiling and caching a function's implementation.
            arg_eval_outs = [arg.eval() for arg in args]

            return_tensors = func(*args, **kwargs)
            if isinstance(return_tensors, Tensor):
                return_tensors = [return_tensors]
            flat_ir = FlatIR(return_tensors)
            G_LOGGER.ir_printer(f"flatIR :\n{flat_ir}")

            with FlatIRCompiler(flat_ir) as executable, FlatIRExecutor(flat_ir) as executor:
                # Create a unique key based on the function's arguments for caching
                cache_key = (args, tuple(sorted(kwargs.items())))
                outputs = executor.execute(*executable)
                tensor_outputs = [Tensor(o) for o in outputs]
                self.cache[cache_key] = tensor_outputs
                if len(self.cache[cache_key]) == 1:
                    return self.cache[cache_key][0]
                else:
                    return self.cache[cache_key]

        return decorated

    def __call__(self, *args, **kwargs):
        if callable(self.func):
            return self.decorated(*args, **kwargs)
        else:
            # args[0] represents the func passed as argument to jit. Ex: jitted_func = jit(func)
            return self._helper(args[0])
