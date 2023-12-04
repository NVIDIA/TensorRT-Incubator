import abc
import cupy
from tripy.ops.storage import Storage
from tripy.common.types import TensorShape

from tripy.common.logging import G_LOGGER

GPU_MEMORY_LIMIT = 1
GPU_MEMORY_SIZE_BYTES = int(GPU_MEMORY_LIMIT * 1024**3)  # convert GB to bytes


class Allocator(abc.ABC):
    @abc.abstractmethod
    def allocate_async(self, s: TensorShape):
        ...


def memory_stats(func):
    def wrap(self, *args, **kwargs):
        G_LOGGER.debug(f"Before Executing {func.__name__}, Memory limit: {self.memory_pool.used_bytes()}")
        try:
            result = func(self, *args, **kwargs)
        except Exception as e:
            G_LOGGER.debug(f"Exception during {func.__name__}: {e}")
            raise
        G_LOGGER.debug(f"After Executing {func.__name__}, Memory limit: {self.memory_pool.used_bytes()}")
        return result

    return wrap


class GpuAllocator(Allocator):
    def __init__(self, memory_limit=GPU_MEMORY_SIZE_BYTES):
        self.memory_limit = memory_limit
        self.stream = cupy.cuda.Stream()
        # Create device type.
        from tripy.common.device import device as make_device

        self.device = make_device("gpu:0")
        # Use default memory pool for now. should we use our own memory allocator?
        self.memory_pool = cupy.get_default_memory_pool()
        self.memory_pool.set_limit(self.memory_limit)

    def __del__(self):
        G_LOGGER.debug(f"Gpu allocator clean up")
        self.memory_pool.free_all_blocks()

    @memory_stats
    def allocate_async(self, s: TensorShape):
        with self.stream:
            try:
                return Storage(
                    self.memory_pool.malloc(s.get_size_in_bytes()), s.get_mlir_dtype(), self.device, s.get_shape_arr()
                )
            except cupy.cuda.memory.OutOfMemoryError:
                G_LOGGER.debug("Error: Out of GPU memory.")
                raise
