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

from nvtripy import export
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.copy import Copy
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
)
def copy(input: "nvtripy.Tensor", device: "nvtripy.device") -> "nvtripy.Tensor":
    r"""
    Returns a copy of the input tensor on the target device.

    Args:
        input: Tensor that will be copied
        device: The target device.

    Returns:
        A copy of input tensor on target device.

    .. code-block:: python
        :linenos:

        input = tp.Tensor([1, 2], device=tp.device("gpu"))
        output = tp.copy(input, tp.device("cpu"))

        assert np.array_equal(np.from_dlpack(output), np.array([1, 2], dtype=np.float32))
        assert output.trace_tensor.producer.device.kind == "cpu"
    """

    return op_utils.create_op(Copy, [input], device)
