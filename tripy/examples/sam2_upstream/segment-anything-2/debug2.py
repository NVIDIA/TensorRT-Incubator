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
# sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
import torch
import tripy as tp
from sam2.modeling.sam.transformer import TwoWayTransformer, TripyTwoWayTransformer

torch.manual_seed(0)
generator = torch.Generator()
generator.manual_seed(0)
t2 = TripyTwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8)
sd = torch.load("checkpoints/sam2_hiera_tiny.pt", map_location="cpu")["model"]

t2_sd = t2.state_dict()
for key in sd:
    if key.startswith("sam_mask_decoder.transformer"):
        new_key = key.replace("sam_mask_decoder.transformer.", "")
        weight = sd[key]
        t2_sd[new_key] = tp.Parameter(weight)

t2.load_from_state_dict(t2_sd)

ie = torch.rand(1, 256, 4, 4, dtype=torch.float32, generator=generator)
ipe = torch.rand(1, 256, 4, 4, dtype=torch.float32, generator=generator)
pe = torch.rand(1, 4, 256, dtype=torch.float32, generator=generator)

print(ie, ipe, pe)

print("***********************************")
print(t2(tp.Tensor(ie), tp.Tensor(ipe), tp.Tensor(pe)))
print("***********************************")
