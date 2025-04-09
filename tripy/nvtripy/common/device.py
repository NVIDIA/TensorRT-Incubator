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
from typing import Any, Dict

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.utils.json import Decoder, Encoder

_VALID_KINDS = {"cpu", "gpu"}


@export.public_api()
@dataclass
class device:
    """
    Represents the device where a tensor will be allocated.

    .. caution:: Using multiple devices is not currently supported, so the device
        index must always be 0.
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
            :caption: First GPU

            gpu_0 = tp.device("gpu:0")

            assert gpu_0.kind == "gpu"
            assert gpu_0.index == 0
        """
        kind, _, index = device.partition(":")
        kind = kind.lower()

        if index:
            try:
                index = int(index)
            except ValueError:
                raise_error(f"Could not interpret: {index} as an integer")
        else:
            index = 0

        if index < 0:
            raise_error(f"Device index must be a non-negative integer, but was: {index}")

        # TODO (#577): Lift this restriction. We will need to check the `Constant` implementation to make sure
        # the allocation happens in the right place. Also check tensor lowering to see that we set the device.
        # NOTE: For CPU, we probably still want to restrict the index to 0.
        if index != 0:
            raise_error(f"Multi-device mode is not currently supported, so device index must be 0, but was: {index}")

        if kind not in _VALID_KINDS:
            raise_error(f"Unrecognized device kind: {kind}. Choose from: {list(_VALID_KINDS)}")

        self.kind = kind
        self.index = index

    # Not putting a docstring so it's not exported. Takes a device name and index directly, sets without validation.
    @staticmethod
    def fast_init(kind: str, index: int) -> "tp.device":
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
