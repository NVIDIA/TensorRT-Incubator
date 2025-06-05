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

__version__ = "0.1.2"

# Import TensorRT to make sure all dependent libraries are loaded first.
import tensorrt

# export.public_api() will expose things here. To make sure that happens, we just need to
# import all the submodules so that the decorator is actually executed.
__all__ = []

import nvtripy


def __discover_modules():
    import importlib
    import pkgutil

    mods = [nvtripy]
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


_ = list(__discover_modules())


def __getattr__(name: str):
    from nvtripy.common.exception import search_for_missing_attr

    look_in = [(nvtripy, "nvtripy")]
    search_for_missing_attr("nvtripy", name, look_in)
