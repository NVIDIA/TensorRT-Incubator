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

import mlir_tensorrt.runtime.api as runtime
from nvtripy.backend.mlir import utils as mlir_utils
from nvtripy.common import device as tp_device
from nvtripy.common.exception import raise_error

EMPTY_MEMREF_CACHE = {}


def create_memref(shape, dtype, device=tp_device("gpu"), stream=None, array=None):
    """
    Creates a memref. If array is provided, it will be populated by the values
    from the array. Otherwise, an uninitialized memref is created.
    """
    is_empty_shape = 0 in shape
    if is_empty_shape:
        cache_key = (tuple(shape), dtype.name, device.kind, device.index)
        if cache_key in EMPTY_MEMREF_CACHE:
            return EMPTY_MEMREF_CACHE[cache_key]

    mlir_dtype = mlir_utils.convert_tripy_dtype_to_runtime_dtype(dtype)

    # "array" is marked as a positional-only argument in MLIR bindings.
    args = [] if array is None else [array]

    kwargs = {"shape": shape, "dtype": mlir_dtype}

    if device.kind == "gpu":
        kwargs["device"] = mlir_utils.MLIRRuntimeClient().get_devices()[device.index]
        # Streams are only allowed for GPU allocations.
        kwargs["stream"] = stream

    memref = mlir_utils.MLIRRuntimeClient().create_memref(*args, **kwargs)

    if is_empty_shape:
        EMPTY_MEMREF_CACHE[cache_key] = memref

    return memref


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
                    return "PyTorch tensors", "tensor.contiguous() or tensor.clone()"
                elif "jaxlib" in obj_type or "jax.numpy" in obj_type:
                    return "JAX arrays", "jax.numpy.asarray(array) or jax.numpy.copy(array)"
                elif "numpy.ndarray" in obj_type:
                    return "NumPy arrays", "np.ascontiguousarray(array) or array.copy(order='C')"
                elif "cupy.ndarray" in obj_type:
                    return "CuPy arrays", "cp.ascontiguousarray(array) or array.copy(order='C')"
                else:
                    return None, None

            tensor_type, contiguous_suggestion = check_tensor_type_and_suggest_contiguous(data)

            error_message = (
                f"Non-canonical strides detected:\n"
                f"  Shape: {shape}\n"
                f"  Strides: {given_strides}\n"
                f"  Expected canonical strides: {canonical_strides}\n"
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
