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
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

from nvtripy import export, utils, wrappers
from nvtripy.frontend.trace.ops import utils as op_utils
from nvtripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Squeeze(BaseTraceOp):

    dims: Tuple[int]

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank - len(self.dims)

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import DynamicReshapeOp

        select_indices = [i for i in range(inputs[0].rank) if i not in self.dims]
        input_shape = op_utils.get_shape_of_tensor(inputs[0])
        shape_slice = []
        for index in select_indices:
            shape_slice.append(op_utils.slice_rank1_tensor(input_shape, index, reason_details=""))

        output_shape = (
            op_utils.concatenate_tensors(shape_slice, dim=0)
            if len(shape_slice) > 0
            else op_utils.add_constant_tensor_from_list([], inputs[0].device)
        )

        DynamicReshapeOp.build([inputs[0], output_shape], outputs)


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int8", "int32", "int64", "bool"]},
)
def squeeze(input: "nvtripy.Tensor", dims: Union[Sequence[int], int]) -> "nvtripy.Tensor":
    """
    Returns a new tensor with all specified singleton dimensions of the input tensor removed.

    Args:
        input: The input tensor.
        dims: The singleton dimension(s) to be removed.
              If this is not provided, all dimensions of size 1 are removed.

    Raises:
        TripyException: If any of the specified dimensions have a size that is not equal to 1.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:
        :caption: Squeeze All Dimensions

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, dims=(0, 2))
        assert np.array_equal(cp.from_dlpack(output).get(), np.squeeze(cp.from_dlpack(input).get()))


    .. code-block:: python
        :linenos:
        :caption: Squeeze First Dimension

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, 0)
        assert np.array_equal(cp.from_dlpack(output).get(), np.squeeze(cp.from_dlpack(input).get(), 0))

    .. code-block:: python
        :linenos:
        :caption: Squeeze First And Third Dimension

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.squeeze(input, (0, 2))

        assert np.array_equal(cp.from_dlpack(output).get(), np.squeeze(cp.from_dlpack(input).get(), (0, 2)))
    """
    return Squeeze.build([input], utils.make_tuple(dims))
