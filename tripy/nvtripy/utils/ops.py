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

from typing import Any, List, Union

from colored import Fore, Style

from nvtripy.common.exception import raise_error
from nvtripy.utils.utils import default


# Like raise_error but adds information about the inputs and output.
def raise_error_io_info(
    op: Union["BaseTraceOp", "BaseFlatIROp"], summary: str, details: List[Any] = None, include_inputs: bool = True
) -> None:
    details = default(details, ["This originated from the following expression:"])
    details += [":"] + op.outputs + ["\n"]
    if include_inputs:
        for index, inp in enumerate(op.inputs):
            details.extend([f"{Fore.magenta}Input {index} was:{Style.reset}", inp])

    raise_error(summary, details)
