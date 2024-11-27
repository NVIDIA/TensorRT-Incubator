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
import tripy as tp
from tripy.frontend import Trace
from tripy.frontend import global_cache
from time import perf_counter

inp = tp.ones((1, 1), dtype=tp.float32)
inp.name = "foo"

layer = tp.Linear(1, 2)
out = layer(inp)

trace = Trace([out])

breakpoint()  # global_cache.get(trace)

start = perf_counter()
out.eval()
print(f"time it took {perf_counter()-start}")

breakpoint()  # global_cache.get(trace)


inp2 = tp.ones((1, 1), dtype=tp.float32)
inp2.name = "father"

layer2 = tp.Linear(1, 2)

out2 = layer2(inp2)
out2.name = "mother"


trace2 = Trace([out2])

breakpoint()  # global_cache.get(trace2)

start = perf_counter()
out2.eval()
print(f"time it took {perf_counter()-start}")

breakpoint()  # global_cache.get(trace2)
