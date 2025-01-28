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
from typing import Any, Dict, List, Tuple

from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Plugin(BaseTraceOp):
    name: str
    version: str
    namespace: str
    output_info: List[Tuple[int, "nvtripy.dtype"]]
    creator_params: Dict[str, Any]

    def infer_dtypes(self):
        for out, (_, dtype) in zip(self.outputs, self.output_info):
            out.dtype = dtype

    def infer_rank(self):
        for out, (rank, _) in zip(self.outputs, self.output_info):
            out.rank = rank

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import PluginOp

        PluginOp.build(inputs, outputs, self.name, self.version, self.namespace, self.creator_params)
