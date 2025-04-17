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
from typing import Any, Dict, List, Sequence, Union

from nvtripy import export, utils
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.plugin import Plugin
from nvtripy.utils.types import str_from_type_annotation, type_str_from_arg


# TODO (pranavm): Add link to custom layers guide once published
@export.public_api(document_under="operations/functions")
def plugin(
    op: str, inputs: Sequence["nvtripy.Tensor"], **kwargs: Dict[str, Any]
) -> Union["nvtripy.Tensor", List["nvtripy.Tensor"]]:
    """
    Calls a TensorRT quickly deployable plugin (QDP).

    Args:
        op: The ID of plugin to call, in the form ``"<namespace>::<name>"``.
        inputs: The inputs to the plugin.
        **kwargs: Additional arguments to pass to the plugin as attributes.
            Supported attribute types are
            `documented here <https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/tensorrt.plugin/trt_plugin_register.html#tensorrt.plugin.register>`_.

    Returns:
        The output(s) of the plugin either as a single tensor if there is only one output,
        or a list of tensors otherwise.
    """
    import tensorrt.plugin as trtp

    if "::" not in op:
        raise_error(f"Invalid plugin ID: '{op}'.", details=["Expected an ID of the form '<namespace>::<name>'."])
    namespace, name = op.split("::")

    try:
        trtp_op = getattr(getattr(trtp.op, namespace), name)
    except AttributeError:
        available_namespaces = [
            ns for ns in (getattr(trtp.op, attr) for attr in dir(trtp.op)) if isinstance(ns, trtp._lib._PluginNamespace)
        ]

        def get_plugins_in_namespace(ns):
            return [attr for attr in dir(ns) if isinstance(getattr(ns, attr), trtp._lib.PluginDef)]

        raise_error(
            f"Plugin '{op}' not found.",
            details=[
                f"Note: Available namespaces and plugins are:\n"
                + "\n".join(
                    [
                        f"- {ns._namespace}::{{ {', '.join(get_plugins_in_namespace(ns))} }}"
                        for ns in available_namespaces
                    ]
                )
            ],
        )

    attrs = {}
    for key, value in kwargs.items():
        if key not in trtp_op.input_attrs:
            raise_error(
                f"Unexpected attribute: '{key}'.",
                details=[f"Available attributes are: {list(trtp_op.input_attrs.keys())}."],
            )

        attr_annotation = trtp_op.input_attrs[key]

        try:
            kwargs[key] = value.tolist()
        except AttributeError:
            if attr_annotation is not type(value):
                raise_error(
                    f"Unexpected attribute type: '{type_str_from_arg(value)}'.",
                    details=[f"For attribute: '{key}', expected type: '{str_from_type_annotation(attr_annotation)}'."],
                )

        attrs[key] = value

    input_descs = [None] * len(inputs)
    for i in range(len(inputs)):
        input_descs[i] = trtp._tensor.TensorDesc()
        input_descs[i].dtype = inputs[i].dtype
        input_descs[i].shape_expr = trtp._tensor.ShapeExprs(inputs[i].rank, _is_dummy=True)
        input_descs[i]._immutable = True
    output_descs = utils.utils.make_tuple(trtp_op.register_func(*input_descs, attrs))

    output_info = [(len(desc.shape_expr), desc.dtype) for desc in output_descs]

    return op_utils.create_op(Plugin, inputs, name, "1", namespace, output_info, kwargs)
