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

from typing import Any

from nvtripy.frontend.constraints.base import Constraints


def doc_str(obj: Any) -> str:
    """
    Returns a string representation of an object for use in the public documentation.
    """
    from nvtripy.common.datatype import dtype as tp_dtype

    if isinstance(obj, tp_dtype):
        return f":class:`{obj.name}`"

    if isinstance(obj, Constraints):
        return obj.doc_str()

    assert False, f"Unsupported object type for doc string generation: {type(obj)}. Please add handling here!"
