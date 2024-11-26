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
__all__ = []


# In order to make the pytest fixtures defined in this submodule visible, we
# need to import them in the test using their function names. To do so, we can
# export them via this file by making them local variables and adding them to `__all__`.
#
# Note that just importing the module is sufficient to update PERF_CASES, but does
# not make the actual fixture function visible to pytest.
def __discover_modules():
    import importlib
    import pkgutil

    mods = [importlib.import_module("tests.performance.cases")]
    while mods:
        mod = mods.pop(0)

        yield mod

        if hasattr(mod, "__path__"):
            mods.extend(
                [
                    importlib.import_module(f"{mod.__name__}.{submod.name}")
                    for submod in pkgutil.iter_modules(mod.__path__)
                ]
            )


modules = list(__discover_modules())[1:]

# Discover and import all perf fixtures.
from tests.performance.conftest import PERF_CASES

__perf_case_names = {case.name for case in PERF_CASES}

for mod in modules:
    for name, obj in mod.__dict__.items():
        if name in __perf_case_names:
            locals()[name] = obj
            __all__.append(name)
