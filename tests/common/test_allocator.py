from typing import List

import ctypes
import numpy as np

from tripy.common.types import TensorShape, MlirDataType
from tripy.common.allocator import GpuAllocator
from tripy.ops.storage import Storage


MEMORY_LIMIT: int = 1 * 1024**3  # 1GB


def get_cuda_device_memory_granularity(device_id: int = 0) -> int:
    """
    Retrieve CUDA memory granularity programmatically.

    Args:
        device_id (int): The CUDA device ID.

    Returns:
        int: CUDA memory granularity.
    """
    return 512


class TestAllocator:
    def test_gpu_allocator(self) -> None:
        """
        Test GPU allocator functionality.
        """
        # Initialize GPU allocator with memory limit
        allocator: GpuAllocator = GpuAllocator(MEMORY_LIMIT)

        # Get CUDA device memory granularity
        granularity: int = get_cuda_device_memory_granularity()

        def round_up(value: int) -> int:
            """
            Round up the given value to the nearest multiple of granularity.

            Args:
                value (int): The value to be rounded up.

            Returns:
                int: Rounded-up value.
            """
            return np.ceil(value / granularity) * granularity

        # Allocate memory in sequence
        elements: List[int] = [1, 512, 10]
        requested_shapes: List[TensorShape] = [TensorShape(MlirDataType(0), [e]) for e in elements]

        allocated_data: List[Storage] = []
        total_requested_bytes: int = 0

        for shape in requested_shapes:
            # Allocate memory using the GPU allocator
            storage: Storage = allocator.allocate_async(shape)

            # Validate the allocated storage
            assert isinstance(storage, Storage)

            # Update total requested bytes
            total_requested_bytes += round_up(shape.get_size_in_bytes())

            # Assert that the memory pool usage reflects the total requested bytes
            assert allocator.memory_pool.used_bytes() >= total_requested_bytes

            # Store the allocated storage
            allocated_data.append(storage)


# Example Usage
if __name__ == "__main__":
    # Create an instance of the test class
    test_instance: TestAllocator = TestAllocator()

    # Run the GPU allocator test
    test_instance.test_gpu_allocator()
