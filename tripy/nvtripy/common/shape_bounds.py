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

from dataclasses import dataclass
from typing import Tuple

from nvtripy import export
from nvtripy.types import IntLike
from nvtripy.utils import json as json_utils


@export.public_api(document_under="compiling_code/input_info", document_init_sig=False)
@dataclass
class ShapeBounds:
    min: Tuple[IntLike]
    """
    The minimum shape.
    """
    opt: Tuple[IntLike]
    """
    The shape to optimize for.
    """
    max: Tuple[IntLike]
    """
    The maximum shape.
    """

    def is_static(self):
        return self.min == self.opt == self.max


@json_utils.Encoder.register(ShapeBounds)
def encode_shape_bounds(shape_bounds):
    return {
        "min": shape_bounds.min,
        "opt": shape_bounds.opt,
        "max": shape_bounds.max,
    }


@json_utils.Decoder.register(ShapeBounds)
def decode_shape_bounds(shape_bounds_dict):
    return ShapeBounds(
        min=tuple(shape_bounds_dict["min"]),
        opt=tuple(shape_bounds_dict["opt"]),
        max=tuple(shape_bounds_dict["max"]),
    )
