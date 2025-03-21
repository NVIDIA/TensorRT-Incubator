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

CONSTANT_IR_PRINT_VOLUME_THRESHOLD = 5
"""
A volume threshold for displaying constants in IR logging messages.
Constants with volumes greater than this threshold will be omitted from logging messages.
"""

STORAGE_OP_CACHE_VOLUME_THRESHOLD = 64
"""
A volume threshold for lifting storage ops to trace inputs for eager mode cache lookups.
Aim is to exclude shape tensors, and TensorRT shape dims supports up to 8 dimensions.
So threshold value should be a lot bigger than 8.
"""
