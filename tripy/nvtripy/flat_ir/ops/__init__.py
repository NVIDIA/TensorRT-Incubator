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

from nvtripy.flat_ir.ops.abs import AbsOp
from nvtripy.flat_ir.ops.add import AddOp
from nvtripy.flat_ir.ops.base import BaseFlatIROp
from nvtripy.flat_ir.ops.broadcast import DynamicBroadcastOp
from nvtripy.flat_ir.ops.clamp import ClampOp
from nvtripy.flat_ir.ops.compare import CompareOp
from nvtripy.flat_ir.ops.concatenate import ConcatenateOp
from nvtripy.flat_ir.ops.constant import ConstantOp
from nvtripy.flat_ir.ops.convert import ConvertOp
from nvtripy.flat_ir.ops.convolution import ConvolutionOp
from nvtripy.flat_ir.ops.copy import CopyOp
from nvtripy.flat_ir.ops.cos import CosineOp
from nvtripy.flat_ir.ops.divide import DivideOp
from nvtripy.flat_ir.ops.dot import DotOp
from nvtripy.flat_ir.ops.exponential import ExpOp
from nvtripy.flat_ir.ops.flip import FlipOp
from nvtripy.flat_ir.ops.floor import FloorOp
from nvtripy.flat_ir.ops.gather import DynamicGatherOp
from nvtripy.flat_ir.ops.get_dimension_size import GetDimensionSizeOp
from nvtripy.flat_ir.ops.iota import DynamicIotaOp
from nvtripy.flat_ir.ops.log import LogOp
from nvtripy.flat_ir.ops.maximum import MaxOp
from nvtripy.flat_ir.ops.minimum import MinOp
from nvtripy.flat_ir.ops.mul import MulOp
from nvtripy.flat_ir.ops.pad import DynamicPadOp
from nvtripy.flat_ir.ops.plugin import PluginOp
from nvtripy.flat_ir.ops.pow import PowOp
from nvtripy.flat_ir.ops.reduce import ArgMinMaxOp, ReduceOp
from nvtripy.flat_ir.ops.reduce_window import ReduceWindowOp
from nvtripy.flat_ir.ops.resize import ResizeCubicOp, ResizeLinearOp, ResizeNearestOp
from nvtripy.flat_ir.ops.reshape import DynamicReshapeOp
from nvtripy.flat_ir.ops.round_nearest_even import RoundNearestEvenOp
from nvtripy.flat_ir.ops.rsqrt import RsqrtOp
from nvtripy.flat_ir.ops.select import SelectOp
from nvtripy.flat_ir.ops.sin import SineOp
from nvtripy.flat_ir.ops.slice import DynamicSliceOp
from nvtripy.flat_ir.ops.sqrt import SqrtOp
from nvtripy.flat_ir.ops.sub import SubtractOp
from nvtripy.flat_ir.ops.tanh import TanhOp
from nvtripy.flat_ir.ops.transpose import TransposeOp
