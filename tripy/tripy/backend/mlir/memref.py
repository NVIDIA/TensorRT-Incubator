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


import mlir_tensorrt.runtime.api as runtime

from tripy.backend.mlir import utils as mlir_utils
from tripy.common import device as tp_device
from tripy.common import utils as common_utils


def create_memref(shape, dtype, device=tp_device("gpu"), stream=None, array=None):
    """
    Creates a memref. If array is provided, it will be populated by the values
    from the array. Otherwise, an empty memref is created.
    """
    mlir_dtype = mlir_utils.convert_tripy_dtype_to_runtime_dtype(dtype)

    args = []

    # "array" is marked as a positional-only argument
    if array is not None:
        args.append(array)

    kwargs = {"shape": shape, "dtype": mlir_dtype}

    if device.kind == "gpu":
        kwargs["device"] = mlir_utils.MLIRRuntimeClient().get_devices()[device.index]
        # Streams are only allowed for GPU allocations.
        kwargs["stream"] = stream

    return mlir_utils.MLIRRuntimeClient().create_memref(*args, **kwargs)


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
    try:
        return memoryview(memref_value).tolist()
    except NotImplementedError as e:
        if "memoryview: format e not supported" in str(e):
            assert memref_value.dtype == runtime.ScalarTypeCode.f16
            return common_utils.Float16MemoryView(bytearray(memref_value)).tolist()
        raise


def pretty_print_memref(memref, threshold=1000, linewidth=10, edgeitems=3):
    """
    Returns a pretty-print string of memref values.
    """
    memref_shape = memref.shape

    def _data_str(data, summarize, linewidth, edgeitems, indent=0):
        if isinstance(data, (float, int)):
            return str(data)

        if len(data) == 0 or isinstance(data[0], (float, int)):
            if summarize and len(data) > 2 * edgeitems:
                data_lines = [data[:edgeitems] + [" ..."] + data[-edgeitems:]]
            else:
                data_lines = [data[i : i + linewidth] for i in range(0, len(data), linewidth)]
            lines = [", ".join([f"{e:.4f}" if isinstance(e, float) else str(e) for e in line]) for line in data_lines]
            return "[" + ("," + "\n" + " " * (indent + 1)).join(lines) + "]"

        if summarize and len(data) > 2 * edgeitems:
            slices = (
                [_data_str(data[i], summarize, linewidth, edgeitems, indent + 1) for i in range(0, edgeitems)]
                + ["..."]
                + [
                    _data_str(data[i], summarize, linewidth, edgeitems, indent + 1)
                    for i in range(len(data) - edgeitems, len(data))
                ]
            )
        else:
            slices = [_data_str(data[i], summarize, linewidth, edgeitems, indent + 1) for i in range(0, len(data))]

        tensor_str = ("," + "\n" * (max(len(memref_shape) - indent - 1, 1)) + " " * (indent + 1)).join(slices)
        return "[" + tensor_str + "]"

    numel = 1
    for d in memref_shape:
        numel *= d
    summarize = numel > threshold
    return _data_str(tolist(memref), summarize, linewidth, edgeitems)
