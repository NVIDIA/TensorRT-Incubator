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
from typing import Any, Dict, Sequence

from nvtripy.utils.json import Decoder, Encoder


# HACK (#109): This is a temporary class which we need in order to convey output information
# to MLIR-TRT. Eventually, it should just call back into Tripy when it needs memory allocated.
@dataclass
class TensorInfo:
    rank: int
    shape: Sequence[int]
    dtype: "nvtripy.dtype"
    device: "nvtripy.device"


@Encoder.register(TensorInfo)
def encode(tensor_info: TensorInfo) -> Dict[str, Any]:
    return {
        "rank": tensor_info.rank,
        "shape": tensor_info.shape,
        "dtype": tensor_info.dtype,
        "device": tensor_info.device,
    }


@Decoder.register(TensorInfo)
def decode(dct: Dict[str, Any]) -> TensorInfo:
    return TensorInfo(dct["rank"], dct["shape"], dct["dtype"], dct["device"])
