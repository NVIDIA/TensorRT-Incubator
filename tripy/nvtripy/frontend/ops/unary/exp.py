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
from nvtripy import export
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.unary import Exp
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def exp(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    r"""
    Computes the elementwise exponential of the elements of the input tensor:

    :math:`\text{exp}(x_{i}) = e^{x_{i}}`

    Args:
        input: The input tensor.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:

        input = tp.arange(3, dtype=tp.float32)
        output = tp.exp(input)

        assert tp.allclose(output, tp.Tensor(np.exp(cp.from_dlpack(input).get())))
    """
    return op_utils.create_op(Exp, [input])
