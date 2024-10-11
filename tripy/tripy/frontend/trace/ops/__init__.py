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

from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.frontend.trace.ops.binary_elementwise import BinaryElementwise, Comparison
from tripy.frontend.trace.ops.cast import Cast
from tripy.frontend.trace.ops.convolution import Convolution
from tripy.frontend.trace.ops.copy import Copy
from tripy.frontend.trace.ops.dequantize import Dequantize
from tripy.frontend.trace.ops.expand import Expand
from tripy.frontend.trace.ops.fill import Fill
from tripy.frontend.trace.ops.flip import Flip
from tripy.frontend.trace.ops.gather import Gather
from tripy.frontend.trace.ops.iota import Iota
from tripy.frontend.trace.ops.matmul import MatrixMultiplication
from tripy.frontend.trace.ops.permute import Permute, Transpose
from tripy.frontend.trace.ops.pad import Pad
from tripy.frontend.trace.ops.plugin import Plugin
from tripy.frontend.trace.ops.quantize import Quantize
from tripy.frontend.trace.ops.reduce import ArgMinMax, Reduce
from tripy.frontend.trace.ops.resize import Resize
from tripy.frontend.trace.ops.reshape import Reshape, Squeeze
from tripy.frontend.trace.ops.shape import Shape
from tripy.frontend.trace.ops.slice import Slice
from tripy.frontend.trace.ops.split import Split
from tripy.frontend.trace.ops.storage import Storage
from tripy.frontend.trace.ops.unary_elementwise import UnaryElementwise
from tripy.frontend.trace.ops.unsqueeze import Unsqueeze
from tripy.frontend.trace.ops.where import Where
