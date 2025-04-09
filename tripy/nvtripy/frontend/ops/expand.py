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
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.broadcast import Broadcast
from nvtripy.types import ShapeLike
from nvtripy.utils import wrappers


def process_sizes(input: "nvtripy.Tensor", sizes: ShapeLike):
    if len(sizes) < input.rank:
        raise_error(
            "The length of `sizes` must be greater or equal to input tensor's rank.",
            [f"sizes has length: {len(sizes)}", f" input rank: {input.rank}"],
        )

    num_prepended = len(sizes) - input.rank
    out_shape = list(sizes[:num_prepended]) + [
        inp_dim if op_utils.is_int_equal_to(out_dim, -1) else out_dim
        for inp_dim, out_dim in zip(input.shape, sizes[num_prepended:])
    ]

    if any(op_utils.is_int_equal_to(dim, -1) for dim in out_shape):
        raise_error(
            "Cannot use -1 for prepended dimension.",
            [
                f"{num_prepended} dimension(s) are going to be prepended since the `sizes` argument "
                f"contains more elements than the number of dimensions in the input.\n"
                f"Prepended dimensions may not contain -1 since there is no corresponding "
                f"dimension in the input to copy from, but got: {sizes}"
            ],
        )

    return {"sizes": out_shape}


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int32", "int64", "bool"],
    },
    convert_to_tensors=True,
    conversion_preprocess_func=process_sizes,
)
def expand(input: "nvtripy.Tensor", sizes: ShapeLike) -> "nvtripy.Tensor":
    """
    Returns a new tensor based on the input tensor with singleton dimensions expanded to a larger size.

    Args:
        input: The input tensor.
        sizes: The desired expanded size.
            A value of :math:`-1` indicates that the dimension should not be modified.
            If the length of this parameter exceeds the rank of the tensor, new dimensions
            are prepended.

    Returns:
        The new tensor.

    .. code-block:: python
        :linenos:

        input = tp.iota((2, 1), dtype=tp.float32)
        output = tp.expand(input, (-1, 4))

        assert np.array_equal(cp.from_dlpack(output).get(), np.broadcast_to(cp.from_dlpack(input).get(), (2, 4)))

    .. code-block:: python
        :linenos:
        :caption: Increasing Tensor Rank

        input = tp.iota((1, 1), dtype=tp.float32)
        output = tp.expand(input, (3, -1, -1))

        assert np.array_equal(cp.from_dlpack(output).get(), np.broadcast_to(cp.from_dlpack(input).get(), (3, 1, 1)))
    """
    from nvtripy.frontend.ops.reshape import reshape

    out_rank = op_utils.get_shape_len(sizes)
    if out_rank > input.rank:
        input = reshape(input, (1,) * (out_rank - input.rank) + input.shape)

    return op_utils.create_op(Broadcast, [input, sizes])
