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


# Here we import only those ops which need to add themselves to the tensor method registry
from nvtripy.frontend.ops.binary.add import __add__
from nvtripy.frontend.ops.binary.div import __rtruediv__, __truediv__
from nvtripy.frontend.ops.binary.mul import __mul__
from nvtripy.frontend.ops.binary.pow import __pow__
from nvtripy.frontend.ops.binary.sub import __rsub__, __sub__
from nvtripy.frontend.ops.binary.floor_div import __floordiv__, __rfloordiv__
from nvtripy.frontend.ops.binary.mod import __mod__, __rmod__
from nvtripy.frontend.ops.binary.less import __lt__
from nvtripy.frontend.ops.binary.equal import __eq__
from nvtripy.frontend.ops.binary.less_equal import __le__
from nvtripy.frontend.ops.binary.not_equal import __ne__
from nvtripy.frontend.ops.binary.greater import __gt__
from nvtripy.frontend.ops.binary.greater_equal import __ge__
from nvtripy.frontend.ops.binary.logical_or import __or__
