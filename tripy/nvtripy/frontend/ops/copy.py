#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvtripy import export
from nvtripy.backend.mlir.utils import MLIRRuntimeClient
from nvtripy.common import device as tp_device
from nvtripy.common.datatype import DATA_TYPES
from nvtripy.common.exception import raise_error
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": list(DATA_TYPES.keys())},
)
def copy(input: "nvtripy.Tensor", device: tp_device) -> "nvtripy.Tensor":
    r"""
    Copies the input tensor to the specified device.

    .. caution:: This function cannot be used in a compiled function or :class:`nvtripy.Module`
        because it depends on evaluating its inputs, which is not allowed during compilation.

    Args:
        input: Input tensor.
        device: The target device to copy the tensor to.

    Returns:
        A new tensor on the specified device.

    Raises:
        TripyException: If the input tensor is already on the specified device, as
            performing copies within the same device is currently not supported.

    .. code-block:: python
        :linenos:
        :caption: Copying To CPU

        input = tp.Tensor([1, 2, 3], device=tp.device("gpu"))
        output = tp.copy(input, device=tp.device("cpu"))

    .. code-block:: python
        :linenos:
        :caption: Copying To GPU

        input = tp.Tensor([1, 2, 3])
        output = tp.copy(input, device=tp.device("gpu"))

    """
    from nvtripy.frontend.tensor import Tensor

    input._eval_for_internal_methods()  # Avoid `eval()` - don't want to inadvertently move the tensor to GPU.
    memref = input.trace_tensor.producer.data
    runtime_client = MLIRRuntimeClient()  # This is a singleton class, so we aren't creating it on each function call.

    # TODO (#577): Support copying between different GPUs:
    if input.device.kind == "cpu" and device.kind == "gpu":
        assert memref.address_space == runtime.PointerType.host
        out_memref = runtime_client.copy_to_device(
            host_memref=memref,
            device=runtime_client.get_devices()[device.index],
        )
    elif input.device.kind == "gpu" and device.kind == "cpu":
        assert memref.address_space == runtime.PointerType.device
        out_memref = runtime_client.copy_to_host(device_memref=memref)
    else:
        raise_error(
            "Copying within the same device kind is not currently supported. Please file an issue if you need this functionality!",
            [
                f"Input tensor has device kind: {input.device.kind}, which is the same kind as the target device: {device}."
            ],
        )

    return Tensor(out_memref)
