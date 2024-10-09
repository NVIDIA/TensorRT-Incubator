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
from typing import Sequence, Union, Tuple
from tripy import export, constraints
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.common.exception import raise_error


@dataclass(repr=False)
class Pad(BaseTraceOp):

    padding_value: Union[int, float]

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank

    def to_flat_ir(self, inputs, outputs):
        from tripy.common.datatype import int32
        from tripy.flat_ir.ops import ConstantOp, DynamicPadOp
        from tripy.flat_ir.tensor import FlatIRTensor

        pad_val_tensor = FlatIRTensor.build(
            shape=(),
            rank=0,
            dtype=outputs[0].dtype,
            device=outputs[0].device,
            reason_details=[f"create the constant value tensor (containing {self.padding_value}) for a pad operation"],
        )
        ConstantOp.build([], [pad_val_tensor], data=self.padding_value)

        # interior_padding is not supported
        # create the default value
        pad_size_shape = (inputs[0].rank,)
        interior_pad_tensor = FlatIRTensor.build(
            shape=pad_size_shape,
            rank=1,
            dtype=int32,
            device=outputs[0].device,
            reason_details=[f"create the default value for interior_padding argument."],
        )
        ConstantOp.build([], [interior_pad_tensor], data=[0] * inputs[0].rank)

        # [operand, pad_val, low, high, interior]
        inputs.insert(1, pad_val_tensor)
        inputs.append(interior_pad_tensor)
        # set padding size tensors' shape
        # because stablehlo requires static shapes
        inputs[2].shape = pad_size_shape
        inputs[3].shape = pad_size_shape
        DynamicPadOp.build(inputs, outputs)


def _convert_pad_sizes(padding_sizes):
    from tripy.common.datatype import int32
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.trace.ops.concatenate import concatenate
    from tripy.frontend.trace.ops.unsqueeze import unsqueeze

    if not any(isinstance(s, Tensor) for s in padding_sizes):
        return Tensor(padding_sizes, dtype=int32)

    sizes_1d = []
    for size in padding_sizes:
        if isinstance(size, Tensor):
            assert size.rank == 0, f"Size expected to be of rank 0, got {size.rank}."
            sizes_1d.append(unsqueeze(size, 0))
        else:
            sizes_1d.append(Tensor([size], dtype=int32))
    return concatenate(sizes_1d, 0)


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bool", "int32"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
def pad(
    input: "tripy.Tensor",
    pad: Sequence[Tuple[Union[int, "tripy.ShapeScalar"]]],
    mode: str = "constant",
    value: Union[int, float] = 0,
) -> "tripy.Tensor":
    r"""
    Pads the input tensor.

    Args:
        input: The input tensor.
        pad: A sequence of padding sizes of each dimension. Its length must be equal to the rank
            of ``input``. Each element of ``pad`` is a tuple of integers or scalars ``(low, high)``,
            which represents the padding sizes before the lowest index and after the highest index at
            the corresponding dimension.
        mode: The padding mode. Only "constant" is supported.
        value: The padding value for "constant" mode.

    Returns:
        The padded tensor.

    .. code-block:: python
        :linenos:
        :caption: Constant padding.

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.pad(input, ((1, 0), (0, 1)))

        input_np = np.arange(6, dtype=np.float32).reshape((2, 3)) # doc: omit
        expected = np.pad(input_np, ((1, 0), (0, 1))) # doc: omit
        assert np.array_equal(cp.from_dlpack(output).get(), expected)
    """
    if len(pad) != input.rank:
        raise_error(
            "`pad` length must equal to the rank of `input`.",
            [f"Got pad={pad}, ", f" input's rank={input.rank}"],
        )

    supported_modes = {"constant"}
    if mode not in supported_modes:
        raise_error(
            "Unsupported padding mode.",
            [f"Got mode={mode}, while supported modes are {supported_modes}"],
        )

    padding_low, padding_high = list(zip(*pad))
    return Pad.build(
        [
            input,
            _convert_pad_sizes(padding_low),
            _convert_pad_sizes(padding_high),
        ],
        value,
    )
