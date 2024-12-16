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

from dataclasses import dataclass
from typing import Sequence, Union

from nvtripy import export, utils, wrappers
from nvtripy.common.exception import raise_error
from nvtripy.frontend.trace.ops import utils as op_utils
from nvtripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Split(BaseTraceOp):
    indices_or_sections: Union[int, Sequence[int]]
    dim: int

    def num_outputs(self):
        if isinstance(self.indices_or_sections, int):
            return self.indices_or_sections
        else:
            # + 1 because of the last split, which is [self.indices_or_sections[-1]:]
            return len(self.indices_or_sections) + 1

    def infer_rank(self):
        for i in range(self.num_outputs()):
            self.outputs[i].rank = self.inputs[0].rank

    def infer_devices(self):
        for i in range(self.num_outputs()):
            self.outputs[i].device = self.inputs[0].device

    def infer_dtypes(self):
        for i in range(self.num_outputs()):
            self.outputs[i].dtype = self.inputs[0].dtype

    # gets input_tensor[..., :,  :, start_idx: end_idx, :, :, ...], with the start and end slice only at the axis dimension
    def build_slice_of_target_dim(self, input_tensor, input_shape, device, start_idx, end_idx, output_tensor):
        from nvtripy.flat_ir.ops import DynamicSliceOp

        start_idxs = []
        limit_idxs = []
        stride_idxs = []

        zero_1d = op_utils.add_constant_tensor_from_list([0], device)
        one_1d = op_utils.add_constant_tensor_from_list([1], device)

        for i in range(input_tensor.rank):
            shape_slice = op_utils.slice_rank1_tensor(
                input_shape,
                i,
                reason_details=[
                    "slicing the shape tensor ",
                    input_shape,
                    f" to get the dimension with index {i}",
                ],
            )

            if i != self.dim:
                start_idxs.append(zero_1d)
                limit_idxs.append(shape_slice)
            else:
                start_idxs.append(start_idx)
                limit_idxs.append(end_idx)
            stride_idxs.append(one_1d)

        start_index_tensor = op_utils.concatenate_tensors(start_idxs, dim=0)
        limit_index_tensor = op_utils.concatenate_tensors(limit_idxs, dim=0)
        stride_index_tensor = op_utils.concatenate_tensors(stride_idxs, dim=0)
        DynamicSliceOp.build(
            [input_tensor, start_index_tensor, limit_index_tensor, stride_index_tensor], [output_tensor]
        )

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.common.datatype import int32
        from nvtripy.flat_ir.ops import DivideOp, MulOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        input_tensor = inputs[0]
        device = input_tensor.device
        input_shape = op_utils.get_shape_of_tensor(input_tensor)
        axis_dim = op_utils.slice_rank1_tensor(
            input_shape, self.dim, reason_details=[f"Getting dim {self.dim} of split input shape {input_shape}"]
        )

        if isinstance(self.indices_or_sections, int):
            section_size_tensor = FlatIRTensor.build(
                shape=[1],
                rank=1,
                dtype=int32,
                device=device,
                reason_details=["Compute size of equal slices"],
            )
            DivideOp.build(
                [axis_dim, op_utils.add_constant_tensor_from_list([self.indices_or_sections], device=device)],
                [section_size_tensor],
            )
            for i in range(self.num_outputs()):
                with FlatIRTensor.context([f"compute indices of split {i}"]):
                    # i*section_size
                    section_i_start_tensor = FlatIRTensor.build(
                        shape=[1],
                        rank=1,
                        dtype=int32,
                        device=device,
                        reason_details=[f"Compute start index"],
                    )
                    MulOp.build(
                        [section_size_tensor, op_utils.add_constant_tensor_from_list([i], device=device)],
                        [section_i_start_tensor],
                    )

                    # (i+1)*section_size
                    section_i_end_tensor = FlatIRTensor.build(
                        shape=[1],
                        rank=1,
                        dtype=int32,
                        device=device,
                        reason_details=[f"Compute end index"],
                    )
                    MulOp.build(
                        [section_size_tensor, op_utils.add_constant_tensor_from_list([i + 1], device=device)],
                        [section_i_end_tensor],
                    )

                    self.build_slice_of_target_dim(
                        input_tensor, input_shape, device, section_i_start_tensor, section_i_end_tensor, outputs[i]
                    )
        else:
            start_index_tensor_split_i = op_utils.add_constant_tensor_from_list([0], device)
            for i, index in enumerate(self.indices_or_sections):
                with FlatIRTensor.context([f"build split for index {index}"]):
                    end_index_tensor_split_i = op_utils.add_constant_tensor_from_list([index], device=device)
                    self.build_slice_of_target_dim(
                        input_tensor,
                        input_shape,
                        device,
                        start_index_tensor_split_i,
                        end_index_tensor_split_i,
                        outputs[i],
                    )
                    start_index_tensor_split_i = end_index_tensor_split_i

            with FlatIRTensor.context(["build final split"]):
                self.build_slice_of_target_dim(
                    input_tensor, input_shape, device, start_index_tensor_split_i, axis_dim, outputs[-1]
                )

    # need override because the default implementation assumes a single output
    def __str__(self) -> str:
        skip_fields = self.str_skip_fields()
        args = [
            f"{field.name}={getattr(self, field.name)}"
            for field in utils.get_dataclass_fields(self, BaseTraceOp)
            if field.name not in skip_fields
        ]

        outputs_string = ", ".join([self.outputs[i].name for i in range(self.num_outputs())])
        return f"{outputs_string} = {self.__class__.__name__.lower()}({', '.join([inp.name for inp in self.inputs] + args)})"


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
)
def split(
    input: "nvtripy.Tensor", indices_or_sections: Union[int, Sequence[int]], dim: int = 0
) -> Union["nvtripy.Tensor", Sequence["nvtripy.Tensor"]]:
    r"""
    Splits `input` along the dimension `dim`, producing slices of the `input` tensor.

    If given a single `int` for `indices_or_sections` (let us call it :math:`n`),
    it produces :math:`n` slices of equal size along dimension `dim` as long as :math:`n` divides the size of dimension `dim`.
    For example, if `input` is one-dimensional and the size of dimension `dim` is :math:`k`, then the result is
    :math:`\texttt{input[:} k/n \texttt{]}`, :math:`\texttt{input[} k/n \texttt{:} 2k/n \texttt{]}`,
    :math:`\ldots`, :math:`\texttt{input[} (n-1)k/n \texttt{:]}`.

    If given a sequence of values for `indices_or_sections`, these will be treated as indices for creating slices.
    For example, if we call the indices :math:`i_0`, :math:`i_1`, :math:`\ldots`, :math:`i_n` and assume `input` is one-dimensional,
    the result is equivalent to :math:`input[:i_0]`, :math:`input[i_0:i_1]`, :math:`input[i_1:i_2]`, :math:`\ldots`, :math:`input[i_n:]`.

    Args:
        input: The input tensor.

        indices_or_sections: If a single integer, it gives the number of equal slices to produce.
            If a list of integers, it gives boundary indices for the slices.

        dim: The dimension along which the slices are done. All other dimensions are included in full.

    Returns:
        A list of slices per the above specification or a single tensor if only one slice is created.

    .. code-block:: python
        :linenos:
        :caption: Simple case.

        input = tp.reshape(tp.arange(16, dtype=tp.float32), (4, 4))
        outputs = tp.split(input, 2, dim=0)
        assert np.array_equal(cp.from_dlpack(outputs[0]).get(), cp.from_dlpack(input[:2, :]).get())
        assert np.array_equal(cp.from_dlpack(outputs[1]).get(), cp.from_dlpack(input[2:, :]).get())

    .. code-block:: python
        :linenos:
        :caption: Choosing a different dimension.

        input = tp.reshape(tp.arange(16, dtype=tp.float32), (4, 4))
        outputs = tp.split(input, 2, dim=1)
        assert np.array_equal(cp.from_dlpack(outputs[0]).get(), cp.from_dlpack(input[:, :2]).get())
        assert np.array_equal(cp.from_dlpack(outputs[1]).get(), cp.from_dlpack(input[:, 2:]).get())

    .. code-block:: python
        :linenos:
        :caption: Multiple index arguments.

        input = tp.reshape(tp.arange(16, dtype=tp.float32), (4, 4))
        outputs = tp.split(input, [1, 2])
        assert np.array_equal(cp.from_dlpack(outputs[0]).get(), cp.from_dlpack(input[:1, :]).get())
        assert np.array_equal(cp.from_dlpack(outputs[1]).get(), cp.from_dlpack(input[1:2, :]).get())
        assert np.array_equal(cp.from_dlpack(outputs[2]).get(), cp.from_dlpack(input[2:, :]).get())
    """
    if dim < 0 or dim >= input.rank:
        raise_error(f"Invalid split dimension {dim}", details=[input])
    if isinstance(indices_or_sections, int):
        if indices_or_sections <= 0:
            raise_error(f"Number of sections argument must be positive, but given {indices_or_sections}")
        num_outputs = indices_or_sections
    else:
        if not indices_or_sections:
            raise_error("Split indices must not be empty")
        last = None
        for index in indices_or_sections:
            if last and index < last:
                raise_error(f"Split indices must be given in ascending order, but given {indices_or_sections}")
            last = index
        num_outputs = len(indices_or_sections) + 1  # add 1 because of the last split

    return Split.build(inputs=[input], indices_or_sections=indices_or_sections, dim=dim, num_outputs=num_outputs)
