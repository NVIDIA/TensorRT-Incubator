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

import json
import typing
from typing import Any, Union

from nvtripy import utils
from nvtripy.utils.json.enc_dec import Decoder, Encoder


def to_json(obj: Any) -> str:
    return json.dumps(obj, cls=Encoder, indent=" " * 4)


def from_json(src: str) -> Any:
    return json.loads(src, object_pairs_hook=Decoder())


def save(obj: Any, dest: Union[str, typing.IO]):
    """
    Saves an object to the specified destination.

    Args:
        obj: The object to save
        dest: A path or file-like object
    """
    utils.utils.save_file(to_json(obj), dest, mode="w")


def load(src: Union[str, typing.IO]) -> Any:
    """
    Loads an object from the specified source.

    Args:
        src: A path or file-like object

    Returns:
        The loaded object
    """
    return from_json(utils.utils.load_file(src, mode="r"))
