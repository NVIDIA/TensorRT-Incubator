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

from typing import Any, Dict, List, Sequence, Tuple, Union

from nvtripy import export
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.plugin import Plugin


@export.public_api(document_under="operations/functions")
def plugin(
    name: str,
    inputs: Sequence["nvtripy.Tensor"],
    output_info: List[Tuple[int, "nvtripy.dtype"]],
    version: str = "1",
    namespace: str = "",
    **kwargs: Dict[str, Any],
) -> Union["Tensor", List["Tensor"]]:
    """
    Calls a TensorRT plugin. Only the ``IPluginV2DynamicExt`` and ``IPluginV3`` interfaces are supported.

    Args:
        name: The name of the plugin to call.
        inputs: The inputs to the plugin.
        output_info: A list of tuples that indicate the rank and data type for each output.
        version: The version of the plugin to call.
        namespace: The namespace of the plugin.
        **kwargs: Additional arguments to pass to the plugin as plugin fields.
            These should be primitive Python types like ``int`` s, ``float`` s, ``str`` s etc.
            Fields that expect ``Dims`` should be provided as a ``tuple`` of ``int`` s.
            Fields that expect multiple values can be provided as ``list`` s or ``tuple`` s.

    Returns:
        The output(s) of the plugin either as a single tensor if there is only one output,
        or a list of tensors otherwise.

    .. code-block:: python
        :linenos:

        inp = tp.iota((2, 1, 4))
        out = tp.plugin(
            name="CustomGeluPluginDynamic",
            inputs=[inp],
            # GELU has a single output which always has the same rank and data
            # type as the input.
            output_info=[(inp.rank, inp.dtype)],
            # The GELU plugin expects a `type_id` parameter indicating the precision
            # to use. `0` indicates float32.
            type_id=0,
        )

        assert tp.allclose(out,tp.gelu(inp),rtol=1e-2) # tp.gelu uses ERF but the plugin uses an approximation.
    """
    return op_utils.create_op(Plugin, inputs, name, version, namespace, output_info, kwargs)
