#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import lru_cache
from typing import Sequence

from tripy.backend.mlir import utils as mlir_utils
from tripy.common import device as tp_device
from tripy.common import utils as common_utils

import mlir_tensorrt.runtime.api as runtime


@lru_cache(maxsize=None)
def _cached_create_empty_memref(shape: Sequence[int], dtype: str, device_kind: str, stream):
    mlirtrt_device = mlir_utils.MLIRRuntimeClient().get_devices()[0] if device_kind == "gpu" else None
    mlirtrt_stream = stream if device_kind == "gpu" else None
    mlir_dtype = mlir_utils.convert_tripy_dtype_to_runtime_dtype(dtype)
    return mlir_utils.MLIRRuntimeClient().create_memref(
        shape=list(shape),
        dtype=mlir_dtype,
        device=mlirtrt_device,
        stream=mlirtrt_stream,
    )


def create_empty_memref(
    shape: Sequence[int],
    dtype: str,
    device: tp_device = tp_device("gpu"),
    stream=None,
    use_cache: bool = True,
):
    """
    Creates an empty memref, used for allocating memory.
    Caches the result for subsequent calls with the same parameters.

    Args:
        use_cache (bool, optional): Whether to use cached results for repeated calls with the same parameters.
                                    If True, returns cached results if available. If False, always creates a new memref.
                                    Defaults to True. This ensures we reuse empty memref across functions.

    """
    if use_cache:
        assert common_utils.is_shape_empty(shape)
        return _cached_create_empty_memref(tuple(shape), dtype, device.kind, stream)
    else:
        return _cached_create_empty_memref.__wrapped__(tuple(shape), dtype, device.kind, stream)


def create_memref_view(data):
    """
    Creates a memref view of an array object that implements the dlpack interface.
    """
    return mlir_utils.MLIRRuntimeClient().create_memref_view_from_dlpack(data.__dlpack__())


# TODO(#134): Consider move below functions to MLIR py bindings
def tolist(memref):
    """
    Converts memref values into a python list.
    """
    memref_value = memref
    if memref.address_space == runtime.PointerType.device:
        memref_value = mlir_utils.MLIRRuntimeClient().copy_to_host(device_memref=memref)
    return memoryview(memref_value).tolist()
