#
# SPDX-FileCopyrightText: Copyright (c) 2024-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Special type annotations used in Tripy.
"""

import numbers
import sys
from typing import Sequence, Union

from nvtripy import export

export.public_api(autodoc_options=[":no-members:", ":no-special-members:"])(sys.modules[__name__])


TensorLike = export.public_api(
    document_under="types.rst",
    module=sys.modules[__name__],
    symbol="TensorLike",
    doc="""
        A :class:`nvtripy.Tensor` or a Python number that can be automatically converted into one.
        """,
)(Union["nvtripy.Tensor", numbers.Number])


IntLike = export.public_api(
    document_under="types.rst",
    module=sys.modules[__name__],
    symbol="IntLike",
    doc="""
        An integer-like object.
        """,
)(Union[int, "nvtripy.DimensionSize"])


ShapeLike = export.public_api(
    document_under="types.rst",
    module=sys.modules[__name__],
    symbol="ShapeLike",
    doc="""
        A shape of a :class:`nvtripy.Tensor` .
        """,
)(Sequence[IntLike])
