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
from sam2.modeling.sam.transformer import TwoWayTransformer, TripyTwoWayTransformer

torch.manual_seed(0)
generator = torch.Generator()
generator.manual_seed(0)

t1 = TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8)

sd = torch.load("checkpoints/sam2_hiera_tiny.pt", map_location="cpu")["model"]

t1_sd = t1.state_dict()
for key in sd:
    if key.startswith("sam_mask_decoder.transformer"):
        new_key = key.replace("sam_mask_decoder.transformer.", "")
        weight = sd[key]
        t1_sd[new_key] = weight

t1.load_state_dict(t1_sd)


ie = torch.rand(1, 256, 4, 4, dtype=torch.float32, generator=generator)
ipe = torch.rand(1, 256, 4, 4, dtype=torch.float32, generator=generator)
pe = torch.rand(1, 4, 256, dtype=torch.float32, generator=generator)

print(ie, ipe, pe)

print("***********************************")
with torch.no_grad():
    print(t1(ie, ipe, pe))
print("***********************************")
