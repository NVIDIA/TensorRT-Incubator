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
from typing import Union, Sequence

from tripy import export

export.public_api()(sys.modules[__name__])

tensor_literal = export.public_api(
    document_under="types.rst",
    autodoc_options=[":no-index:"],
    module=sys.modules[__name__],
    symbol="tensor_literal",
)(Union[numbers.Number, Sequence["tripy.types.tensor_literal"]])

tensor_literal.__doc__ = """Denotes the recursive type annotation for tensor literals.
A tensor literal can be a Python number or a sequence of tensor literals
(i.e., a sequence of numbers of any depth).
"""
