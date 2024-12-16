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

from nvtripy.frontend.trace.ops.base import BaseTraceOp
from nvtripy.frontend.trace.ops.binary_elementwise import BinaryElementwise, Comparison
from nvtripy.frontend.trace.ops.cast import Cast
from nvtripy.frontend.trace.ops.convolution import Convolution
from nvtripy.frontend.trace.ops.copy import Copy
from nvtripy.frontend.trace.ops.dequantize import Dequantize
from nvtripy.frontend.trace.ops.expand import Expand
from nvtripy.frontend.trace.ops.fill import Fill
from nvtripy.frontend.trace.ops.flip import Flip
from nvtripy.frontend.trace.ops.gather import Gather
from nvtripy.frontend.trace.ops.iota import Iota
from nvtripy.frontend.trace.ops.matmul import MatrixMultiplication
from nvtripy.frontend.trace.ops.pad import Pad
from nvtripy.frontend.trace.ops.permute import Permute
from nvtripy.frontend.trace.ops.plugin import Plugin
from nvtripy.frontend.trace.ops.quantize import Quantize
from nvtripy.frontend.trace.ops.reduce import ArgMinMax, Reduce
from nvtripy.frontend.trace.ops.reshape import Reshape
from nvtripy.frontend.trace.ops.resize import Resize
from nvtripy.frontend.trace.ops.shape import GetDimensionSize
from nvtripy.frontend.trace.ops.slice import Slice
from nvtripy.frontend.trace.ops.split import Split
from nvtripy.frontend.trace.ops.squeeze import Squeeze
from nvtripy.frontend.trace.ops.storage import Storage
from nvtripy.frontend.trace.ops.unary_elementwise import UnaryElementwise
from nvtripy.frontend.trace.ops.where import Where
