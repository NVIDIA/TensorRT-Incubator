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
from tests import helper

import nvtripy as tp


class TestLayerNorm:

    def test_layernorm_improper_dimensions(self):
        tp_layernorm = tp.LayerNorm(
            normalized_shape=[2, 2],
        )
        tp_layernorm.initialize_dummy_parameters()

        x = tp.ones((5, 5, 5))
        with helper.raises(
            tp.TripyException, match="The normalization scale is not broadcast-compatible with the input at dimension 1"
        ):
            tp_layernorm(x).eval()

    def test_layernorm_improper_rank(self):
        tp_layernorm = tp.LayerNorm(
            normalized_shape=[2],
        )
        tp_layernorm.initialize_dummy_parameters()

        x = tp.ones((2,))
        with helper.raises(
            tp.TripyException,
            match=f"Input must have a rank of at least 2, but got input of rank: {x.rank}",
        ):
            tp_layernorm(x).eval()
