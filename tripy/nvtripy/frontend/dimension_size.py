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

from typing import Optional, Union

from nvtripy import export
from nvtripy.common.datatype import int32
from nvtripy.frontend.tensor import Tensor


@export.public_api(document_under="tensor")
class DimensionSize(Tensor):
    """
    A 0D, :class:`int32` tensor that represents a scalar value extracted from the shape of a tensor.
    """

    def __init__(self, data: int, name: Optional[str] = None) -> None:
        r"""
        Args:
            data: The value of the DimensionSize, which should be a scalar integer.
            name: An optional name.
        """
        super().__init__(data=data, dtype=int32, name=name)

    def __int__(self) -> int:
        return self.tolist()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        val = self.tolist()
        assert isinstance(val, int)
        return str(val)

    def eval(self) -> "nvtripy.Tensor":
        from nvtripy.trace.ops.shape import GetDimensionSize, Shape

        # TODO (#593): Generalize this to any branchy graph:
        # If we find a pattern like Shape -> GetDimensionSize, we want to eval the Shape operation
        # so that we aren't evaluating the entire graph for each dimension.
        producer = self.trace_tensor.producer
        if isinstance(producer, GetDimensionSize) and isinstance(producer.inputs[0].producer, Shape):
            frontend_tensor = producer.inputs[0].frontend_tensor
            frontend_tensor.eval()

            dim_size = GetDimensionSize([frontend_tensor.trace_tensor], dim=producer.dim)
            dim_size.outputs[0].is_compile_tracer = self.trace_tensor.is_compile_tracer
            self.trace_tensor = dim_size.outputs[0]

        return super().eval()
