import os
import numpy as np
from typing import List

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))

# Supported NumPy data types
NUMPY_TYPES = [
    np.int8,
    # np.int16,
    np.int32,
    # (38): Add cast operation to support unsupported backend types.
    # np.int64, # TODO: Fails as convert-hlo-to-tensorrt=allow-i64-to-i32-conversion is enabled in the mlir backend. Explore disabling - limitation can due to TRT limitation.
    np.uint8,
    # np.uint16,
    # np.uint32,
    # np.uint64,
    np.float16,
    np.float32,
    # (38): Add cast operation to support unsupported backend types.
    # np.float64, # TODO: How do we support in tripy? May be insert a cast to float32 in case of lossless conversion and error otherwise?
]


def torch_type_supported(data: np.ndarray):
    unsupported_dtypes = [np.uint16, np.uint32, np.uint64]
    return data.dtype not in unsupported_dtypes


def all_same(a: List[int] or List[float], b: List[int] or List[float]):
    """
    Compare two lists element-wise for equality.

    Args:
        a (list): The first list.
        b (list): The second list.

    Returns:
        bool: True if the lists have the same elements in the same order, False otherwise.
    """
    if len(a) != len(b):
        return False

    for a, b in zip(a, b):
        if a != b:
            return False

    return True
