import atexit
import functools
import glob
import os
import tempfile
from collections import namedtuple
from typing import Callable, Dict, Tuple

from tripy import config, utils
from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor
from tripy.common.logging import G_LOGGER
from tripy.frontend import Tensor, nn
from tripy.frontend.trace import Trace

JITValue = namedtuple("JITValue", ["executable", "flat_ir_shape_info"])


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

            print(out)
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

            print(out)
            assert (out.numpy() == np.array([2.0, 2.0])).all()
        """
        self.kwargs = kwargs
        self.cache: Dict[str, JITValue] = {}
        self._const_args: Tuple[int] = ()
        self._func: Callable = None
        self._decorated: Callable = None
        self._obj = None
        if func is not None:
            self._func = func
            self._decorated = self._helper(func)
        atexit.register(self.destroy)

        if "cache_dir" in kwargs:
            self.cache_dir = kwargs["cache_dir"]
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            elif len(os.listdir(self.cache_dir)):
                self.load(self.cache_dir)
        else:
            os.makedirs(config.JIT_CACHE_DIR, exist_ok=True)
            self.cache_dir = tempfile.mkdtemp(dir=config.JIT_CACHE_DIR)

    def __get__(self, obj, type=None):
        self._obj = obj
        return self

    def _can_reuse_cached_executable(self, cache_key, flat_ir):
        # Return true if cached inputs shapes are super set of current inputs shapes.
        cached_inputs, _ = self.cache[cache_key].flat_ir_shape_info
        curr_inputs, _ = flat_ir.io_shape_info()
        return all(map(lambda shapes: shapes[0].is_a_subset_of(shapes[1]), zip(curr_inputs, cached_inputs)))

    def _get_jit_hash(self, trace):
        from tripy import __version__ as tp_version
        from tripy.frontend.dim import Dim

        hashable_flat_ir = trace.to_flat_ir()
        i_info, o_info = hashable_flat_ir.io_tensor_info()

        def to_dynamic(shape):
            return [Dim(-1) for _ in shape]

        for i in range(len(i_info)):
            hashable_flat_ir.inputs[i].shape = to_dynamic(i_info[i].shape)

        for l in hashable_flat_ir.ops:
            for i in range(len(l.inputs)):
                l.inputs[i].shape = to_dynamic(l.inputs[i].shape)
            for o in range(len(l.outputs)):
                l.outputs[o].shape = to_dynamic(l.outputs[o].shape)

        for o in range(len(o_info)):
            hashable_flat_ir.outputs[o].shape = to_dynamic(o_info[o].shape)

        G_LOGGER.debug(f"Hashable FlatIR :\n{hashable_flat_ir}")

        # TODO: improve the naive ir hash
        hash_str = str(hash(tp_version + str(hashable_flat_ir)))
        return hash_str.zfill(config.JIT_CACHE_HASH_LENGTH)

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
                    eval_args[i].op.const_fold = False
            eval_args = tuple(eval_args)

            if self._obj is not None:
                return_tensors = func(self._obj, *eval_args, **kwargs)
            else:
                return_tensors = func(*eval_args, **kwargs)
            if isinstance(return_tensors, Tensor):
                return_tensors = [return_tensors]
            trace = Trace(return_tensors)

            G_LOGGER.ir_printer(f"Trace :\n{trace}")
            flat_ir = trace.to_flat_ir()
            G_LOGGER.ir_printer(f"FlatIR :\n{flat_ir}")
            output_devices = [o.device for o in flat_ir.outputs]
            cache_key = self._get_jit_hash(trace)

            if cache_key in self.cache and self._can_reuse_cached_executable(cache_key, flat_ir):
                executable = self.cache[cache_key].executable
            else:
                compiler = FlatIRCompiler()
                executable = compiler.compile(flat_ir)
                self.cache[cache_key] = JITValue(executable, flat_ir.io_shape_info())

            i_tensor_info, o_tensor_info = flat_ir.io_tensor_info()
            executor = FlatIRExecutor(executable, output_devices, i_tensor_info, o_tensor_info)
            outputs = executor.execute(eval_args)
            tensor_outputs = [
                Tensor(o.data.view(), device=out_device) for o, out_device in zip(outputs, executor.output_devices)
            ]
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

        self.save(self.cache_dir)
        mlir_backend = mlir_wrapper()
        for _, value in self.cache.items():
            mlir_backend.exec_destroy(value.executable)

    def save(self, dir_path=None):
        """
        Saves all cached executables to a given folder

        Args:
            dir_path: A string of folder name

        Example:
        ::

            import tempfile
            import numpy as np

            a = tp.Tensor([1.0, 1.0], dtype=tp.float32, device=tp.device("gpu"))
            b = tp.Tensor([1.0, 1.0], dtype=tp.float32, device=tp.device("gpu"))

            @tp.jit
            def add(a, b):
                c = a + b
                return c

            out = add(a, b)
            with tempfile.TemporaryDirectory() as tmp_dir:
                add.save(tmp_dir)
                add.load(tmp_dir)
        """
        from tripy.backend.mlir.mlir import mlir_wrapper
        import pickle

        G_LOGGER.info(f"Saving engines to cache: {dir_path}")
        mlir_backend = mlir_wrapper()
        dir_path = utils.default(dir_path, self.cache_dir)
        os.makedirs(dir_path, exist_ok=True)

        for index, (cache_key, cache_value) in enumerate(self.cache.items()):
            filename = os.path.join(dir_path, f"engine_{index}.engine")
            if os.path.exists(filename):
                os.remove(filename)
            with open(filename, "wb") as f:
                f.write(cache_key.encode())
                pickle.dump(cache_value.flat_ir_shape_info, f)
            mlir_backend.save(cache_value.executable, filename)

    def load(self, dir_path=None):
        """
        Loads all compiled executables from a given directory

        Args:
            dir_path: A string of folder name
        """
        from tripy.backend.mlir.mlir import mlir_wrapper
        import pickle

        G_LOGGER.info(f"Loading engines from cache: {dir_path}")
        mlir_backend = mlir_wrapper()
        dir_path = utils.default(dir_path, self.cache_dir)
        if not os.path.exists(dir_path):
            raise Exception("Folder does not exist!")

        for engine_name in glob.iglob(os.path.join(dir_path, "*.engine")):
            engine_name = os.path.join(dir_path, engine_name)
            with open(engine_name, "rb") as f:
                cache_key = f.read(config.JIT_CACHE_HASH_LENGTH).decode()
                flat_ir_shape_info = pickle.load(f)
                executable = mlir_backend.load(data=f.read())
                self.cache[cache_key] = JITValue(executable, flat_ir_shape_info)
