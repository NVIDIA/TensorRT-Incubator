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
import numbers
from typing import Union

from nvtripy import export
from nvtripy.common import datatype
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops.cast import cast
from nvtripy.frontend.ops.reshape import reshape
from nvtripy.trace.ops.linspace import Linspace
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"dtype": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
)
def arange(
    start: Union[numbers.Number, "nvtripy.DimensionSize"],
    stop: Union[numbers.Number, "nvtripy.DimensionSize"],
    step: Union[numbers.Number, "nvtripy.DimensionSize"] = 1,
    dtype: "nvtripy.dtype" = datatype.float32,
) -> "nvtripy.Tensor":
    r"""
    Returns a 1D tensor containing a sequence of numbers in the half-open interval
    :math:`[\text{start}, \text{stop})` incrementing by :math:`\text{step}`.

    Args:
        start: The inclusive lower bound of the values to generate. If a tensor is provided, it must be a scalar tensor.
        stop: The exclusive upper bound of the values to generate. If a tensor is provided, it must be a scalar tensor.
        step: The spacing between values. If a tensor is provided, it must be a scalar tensor.
        dtype: The desired data type of the tensor.

    Returns:
        A tensor of shape :math:`[\frac{\text{stop}-\text{start}}{\text{step}}]`.

    .. code-block:: python
        :linenos:

        output = tp.arange(0.5, 2.5)

        assert (cp.from_dlpack(output).get() == np.arange(0.5, 2.5, dtype=np.float32)).all()

    .. code-block:: python
        :linenos:
        :caption: Custom ``step`` Value

        output = tp.arange(2.3, 0.8, -0.2)

        assert tp.allclose(output, tp.Tensor(np.arange(2.3, 0.8, -0.2, dtype=np.float32)))
    """
    from nvtripy.frontend.dimension_size import DimensionSize
    from nvtripy.frontend.tensor import Tensor

    if isinstance(step, numbers.Number) and step == 0:
        raise_error("Step in arange cannot be 0.")

    # math.ceil(a / b) is same as -(-a // b). Don't use math.ceil as start, stop or step can be Tensor.
    size = 0 - ((start - stop) // step)
    if isinstance(size, numbers.Number) and size <= 0:
        raise_error(
            "Arange tensor is empty.",
            details=[
                f"start={start}, stop={stop}, step={step}",
            ],
        )

    if not isinstance(size, DimensionSize):
        size = int(size)

    size = op_utils.tensor_from_shape_like([size])

    linspace_dtype = Linspace.get_closest_dtype(dtype)
    start = Tensor(start, dtype=linspace_dtype) if not isinstance(start, DimensionSize) else cast(start, linspace_dtype)
    step = (
        Tensor([step], dtype=linspace_dtype)
        if not isinstance(step, DimensionSize)
        else cast(reshape(step, (1,)), linspace_dtype)
    )
    out = op_utils.create_op(Linspace, [size, start, step], dtype=linspace_dtype)
    return cast(out, dtype)


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"dtype": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
)
def arange(
    stop: Union[numbers.Number, "nvtripy.DimensionSize"], dtype: "nvtripy.dtype" = datatype.float32
) -> "nvtripy.Tensor":
    r"""
    Returns a 1D tensor containing a sequence of numbers in the half-open interval
    :math:`[0, \text{stop})` incrementing by 1.


    Args:
        stop: The exclusive upper bound of the values to generate.
        dtype: The desired datatype of the tensor.

    Returns:
        A tensor of shape :math:`[\text{stop}]`.

    .. code-block:: python
        :linenos:

        output = tp.arange(5)

        assert (cp.from_dlpack(output).get() == np.arange(5, dtype=np.float32)).all()
    """
    return arange(0, stop, dtype=dtype)
