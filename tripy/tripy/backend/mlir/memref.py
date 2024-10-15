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

import re

from functools import lru_cache
from typing import Sequence

from tripy.utils import raise_error
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
    device: tp_device = tp_device(("gpu", 0)),
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
    try:
        memref = mlir_utils.MLIRRuntimeClient().create_memref_view_from_dlpack(
            data.__dlpack__(), assert_canonical_strides=True
        )
    except runtime.MTRTException as e:
        error_msg = str(e)
        match = re.search(
            r"Given strides \[([\d, ]+)\] do not match canonical strides \[([\d, ]+)\] for shape \[([\d, ]+)\]",
            error_msg,
        )

        if match:
            given_strides = [int(s) for s in match.group(1).split(",")]
            canonical_strides = [int(s) for s in match.group(2).split(",")]
            shape = [int(s) for s in match.group(3).split(",")]

            def check_tensor_type_and_suggest_contiguous(obj):
                obj_type = str(type(obj))
                if "torch.Tensor" in obj_type:
                    return "PyTorch Tensor", "tensor.contiguous() or tensor.clone()"
                elif "jaxlib" in obj_type or "jax.numpy" in obj_type:
                    return "JAX Array", "jax.numpy.asarray(array) or jax.numpy.copy(array)"
                elif "numpy.ndarray" in obj_type:
                    return "NumPy Array", "np.ascontiguousarray(array) or array.copy(order='C')"
                elif "cupy.ndarray" in obj_type:
                    return "CuPy Array", "cp.ascontiguousarray(array) or array.copy(order='C')"
                else:
                    return None, None

            tensor_type, contiguous_suggestion = check_tensor_type_and_suggest_contiguous(data)

            error_message = (
                f"Non-canonical strides detected:\n"
                f"  Shape: {shape}\n"
                f"  Current stride: {given_strides}\n"
                f"  Expected canonical stride: {canonical_strides}\n"
                f"Non-canonical strides are not supported for Tripy tensors. "
                f"This usually occurs when the tensor is not contiguous in memory. "
                + (
                    f"To resolve this issue:\n"
                    f"For {tensor_type}, use {contiguous_suggestion} to ensure contiguity before converting to a Tripy tensor."
                    if tensor_type is not None
                    else ""
                )
            )
            raise_error(error_message)
        else:
            # If the error message doesn't match the expected format, re-raise the original exception
            raise
    return memref


# TODO(#134): Consider move below functions to MLIR py bindings
def tolist(memref):
    """
    Converts memref values into a python list.
    """
    memref_value = memref
    if memref.address_space == runtime.PointerType.device:
        memref_value = mlir_utils.MLIRRuntimeClient().copy_to_host(device_memref=memref)
    return memoryview(memref_value).tolist()
