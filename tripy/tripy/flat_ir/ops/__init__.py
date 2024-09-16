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

from tripy.flat_ir.ops.abs import AbsOp
from tripy.flat_ir.ops.add import AddOp
from tripy.flat_ir.ops.base import BaseFlatIROp
from tripy.flat_ir.ops.broadcast import DynamicBroadcastOp
from tripy.flat_ir.ops.clamp import ClampOp
from tripy.flat_ir.ops.compare import CompareOp
from tripy.flat_ir.ops.concatenate import ConcatenateOp
from tripy.flat_ir.ops.constant import ConstantOp
from tripy.flat_ir.ops.convert import ConvertOp
from tripy.flat_ir.ops.convolution import ConvolutionOp
from tripy.flat_ir.ops.copy import CopyOp
from tripy.flat_ir.ops.cos import CosineOp
from tripy.flat_ir.ops.divide import DivideOp
from tripy.flat_ir.ops.dot import DotOp
from tripy.flat_ir.ops.exponential import ExpOp
from tripy.flat_ir.ops.flip import FlipOp
from tripy.flat_ir.ops.floor import FloorOp
from tripy.flat_ir.ops.gather import DynamicGatherOp
from tripy.flat_ir.ops.get_dimension_size import GetDimensionSizeOp
from tripy.flat_ir.ops.iota import DynamicIotaOp
from tripy.flat_ir.ops.log import LogOp
from tripy.flat_ir.ops.maximum import MaxOp
from tripy.flat_ir.ops.minimum import MinOp
from tripy.flat_ir.ops.mul import MulOp
from tripy.flat_ir.ops.plugin import PluginOp
from tripy.flat_ir.ops.pow import PowOp
from tripy.flat_ir.ops.reduce import ArgMinMaxOp, ReduceOp
from tripy.flat_ir.ops.reduce_window import ReduceWindowOp
from tripy.flat_ir.ops.reshape import DynamicReshapeOp
from tripy.flat_ir.ops.round_nearest_even import RoundNearestEvenOp
from tripy.flat_ir.ops.rsqrt import RsqrtOp
from tripy.flat_ir.ops.select import SelectOp
from tripy.flat_ir.ops.sin import SineOp
from tripy.flat_ir.ops.slice import DynamicSliceOp
from tripy.flat_ir.ops.sqrt import SqrtOp
from tripy.flat_ir.ops.sub import SubtractOp
from tripy.flat_ir.ops.tanh import TanhOp
from tripy.flat_ir.ops.transpose import TransposeOp
