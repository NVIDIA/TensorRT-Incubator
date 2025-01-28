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

from nvtripy.trace.ops.base import BaseTraceOp
from nvtripy.trace.ops.binary_elementwise import BinaryElementwise, Comparison
from nvtripy.trace.ops.cast import Cast
from nvtripy.trace.ops.convolution import Convolution
from nvtripy.trace.ops.copy import Copy
from nvtripy.trace.ops.dequantize import Dequantize
from nvtripy.trace.ops.expand import Expand
from nvtripy.trace.ops.fill import Fill
from nvtripy.trace.ops.flip import Flip
from nvtripy.trace.ops.gather import Gather
from nvtripy.trace.ops.iota import Iota
from nvtripy.trace.ops.matmul import MatrixMultiplication
from nvtripy.trace.ops.pad import Pad
from nvtripy.trace.ops.permute import Permute
from nvtripy.trace.ops.plugin import Plugin
from nvtripy.trace.ops.quantize import Quantize
from nvtripy.trace.ops.reduce import ArgMinMax, Reduce
from nvtripy.trace.ops.reshape import Reshape
from nvtripy.trace.ops.resize import Resize
from nvtripy.trace.ops.shape import GetDimensionSize
from nvtripy.trace.ops.slice import Slice
from nvtripy.trace.ops.split import Split
from nvtripy.trace.ops.squeeze import Squeeze
from nvtripy.trace.ops.storage import Storage
from nvtripy.trace.ops.unary_elementwise import UnaryElementwise
from nvtripy.trace.ops.where import Where
