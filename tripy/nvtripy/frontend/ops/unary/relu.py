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

from nvtripy import export
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.unary import Relu
from nvtripy.utils import wrappers
from nvtripy.common import datatype


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int4", "int32", "int64", "int8"],
    },
)
def relu(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    r"""
    Applies Rectified Linear Unit (RELU) function
    to each element of the input tensor:

    :math:`\text{relu}(x) = \max(0,x)`

    Args:
        input: The input tensor.

    Returns:
        A tensor of the same shape as the input.

    .. code-block:: python
        :linenos:

        input = tp.Tensor([1., 2., 3., 4.], dtype=tp.float32)
        output = tp.relu(input)

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # doc: omit
        assert tp.allclose(output, tp.Tensor(torch.nn.functional.relu(t)))

    """
    from nvtripy.frontend.ops.binary.maximum import maximum
    from nvtripy.frontend.tensor import Tensor

    if issubclass(input.dtype, datatype.integer):
        # Activation in TensorRT does not support integral types.
        return maximum(input, Tensor(0, dtype=input.dtype))

    return op_utils.create_op(Relu, [input])
