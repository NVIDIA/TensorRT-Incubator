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

from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.utils import utils


def transform_pooling_params(kernel_dims, stride, padding):
    spatial_dims = len(kernel_dims)
    if spatial_dims != 2 and spatial_dims != 3:
        raise_error("Unsupported kernel_dims, must be 2D or 3D.", [f"Got kernel_dims={kernel_dims}"])

    stride = utils.default(stride, [1] * spatial_dims)
    padding = utils.default(padding, [(0, 0)] * spatial_dims)
    pre_padding, post_padding = op_utils.transform_conv_pooling_padding(padding)
    return stride, pre_padding, post_padding
