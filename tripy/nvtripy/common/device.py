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
from typing import Any, Dict

from nvtripy import export
from nvtripy.common.exception import TripyException
from nvtripy.utils.json import Decoder, Encoder

_VALID_KINDS = {"cpu", "gpu"}


@export.public_api()
@dataclass
class device:
    # TODO: Improve docstrings here. Unclear what other information we'd want to include.
    """
    Represents the device where a tensor will be allocated.
    """

    kind: str
    index: int

    def __init__(self, device: str) -> None:
        r"""
        Args:
            device: A string consisting of the device kind and an optional index.
                    The device kind may be one of: ``["cpu", "gpu"]``.
                    If the index is provided, it should be separated from the device kind
                    by a colon (``:``). If the index is not provided, it defaults to 0.

        .. code-block:: python
            :linenos:
            :caption: Default CPU

            cpu = tp.device("cpu")

            assert cpu.kind == "cpu"
            assert cpu.index == 0

        .. code-block:: python
            :linenos:
            :caption: Second GPU

            gpu_1 = tp.device("gpu:1")

            assert gpu_1.kind == "gpu"
            assert gpu_1.index == 1
        """
        kind, _, index = device.partition(":")
        kind = kind.lower()

        if index:
            try:
                index = int(index)
            except ValueError:
                raise TripyException(f"Could not interpret: {index} as an integer")
        else:
            index = 0

        if index < 0:
            raise TripyException(f"Device index must be a non-negative integer, but was: {index}")

        if kind not in _VALID_KINDS:
            raise TripyException(f"Unrecognized device kind: {kind}. Choose from: {list(_VALID_KINDS)}")

        self.kind = kind
        self.index = index

    # Not putting a docstring so it's not exported. Takes a device name and index directly, sets without validation.
    @staticmethod
    def create_directly(kind: str, index: int) -> "tp.device":
        instance = device.__new__(device)
        instance.kind = kind
        instance.index = index
        return instance

    def __str__(self) -> str:
        return f"{self.kind}:{self.index}"


@Encoder.register(device)
def encode(dev: device) -> Dict[str, Any]:
    return {"kind": dev.kind, "index": dev.index}


@Decoder.register(device)
def decode(dct: Dict[str, Any]) -> device:
    return device(f"{dct['kind']}:{dct['index']}")
