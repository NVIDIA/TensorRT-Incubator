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

import copy
from dataclasses import dataclass, field
from typing import Dict, List

from colored import Fore


@dataclass
class VerbosityConfig:
    prefix: str
    color: str
    enables: List[str] = field(default_factory=list)
    """
    Name(s) of other verbosities that this verbosity will automatically enable.
    """


def make_verbosity_map() -> Dict[str, VerbosityConfig]:
    verbosity_map = {
        "verbose": VerbosityConfig("[V] ", Fore.light_magenta, ["info"]),
        "info": VerbosityConfig("[I] ", "", ["warning"]),
        "warning": VerbosityConfig("[W] ", Fore.light_yellow, ["error"]),
        "error": VerbosityConfig("[E] ", Fore.light_red),
        "trace": VerbosityConfig("==== Trace IR ====\n", Fore.magenta),
        "mlir": VerbosityConfig("==== MLIR ====\n", Fore.magenta),
        # Shorthand for enabling all IR dumps,
        # `logger.ir` probably shouldn't be called but `"ir"` may be used as a verbosity option.
        "ir": VerbosityConfig("", "", enables=["trace", "mlir"]),
        "timing": VerbosityConfig("==== Timing ====\n", Fore.cyan),
    }

    # Do a pass to recursively expand enables
    for verbosity in verbosity_map.values():
        new_enables = copy.copy(verbosity.enables)

        index = 0
        while index < len(new_enables):
            new_enables.extend(verbosity_map[new_enables[index]].enables)
            index += 1

        verbosity.enables = list(set(new_enables))

    return verbosity_map
