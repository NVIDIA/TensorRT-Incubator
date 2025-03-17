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

from typing import List, Sequence, Tuple, Union

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.plugin import Plugin
from nvtripy.utils.types import str_from_type_annotation, type_str_from_arg
import tensorrt.plugin as trtp
import numpy as np

@export.public_api(document_under="operations/functions")
def plugin(
    name: str,
    inputs: Sequence["nvtripy.Tensor"],
    output_info: List[Tuple[int, "nvtripy.dtype"]],
    version: str = "1",
    namespace: str = "",
    **kwargs,
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
            "CustomGeluPluginDynamic",
            [inp],
            # GELU has a single output which always has the same rank and data
            # type as the input.
            output_info=[(inp.rank, inp.dtype)],
            # The GELU plugin expects a `type_id` parameter indicating the precision
            # to use. `0` indicates float32.
            type_id=0,
        )

        assert tp.allclose(out,tp.gelu(inp),rtol=1e-2) # tp.gelu uses ERF but the plugin uses an approximation.
    """
    return op_utils.create_op(
        Plugin,
        inputs,
        name,
        version,
        namespace,
        output_info,
        kwargs,
    )

@export.public_api(document_under="operations/functions")
def plugin(
    op: str,
    inputs: Sequence["nvtripy.Tensor"],
    **kwargs,
) -> Union["Tensor", List["Tensor"]]:
    """
    Calls a TensorRT quickly deployable plugin (QDP).

    Args:
        op: The id of plugin to call. Should be of the form ""<namespace>::<name>"".
        inputs: The inputs to the plugin.
        **kwargs: Additional arguments to pass to the plugin as plugin fields.
            Supported types are the same as those supported by TensorRT QDPs:
            (1). Scalar Python primitives like ``int`` s, ``float`` s, ``str`` s etc.
            (2). 1-D Numpy arrays.

    Returns:
        The output(s) of the plugin either as a single tensor if there is only one output,
        or a list of tensors otherwise.

    .. code-block:: python
        :linenos:

        import triton
        import triton.language as tl

        import tensorrt.plugin as trtp

        @triton.jit
        def add_kernel(x_ptr, n_elements, y_ptr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(y_ptr + offsets, x + 1, mask=mask)

        @trtp.register("example::elemwise_add_plugin")
        def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> trtp.TensorDesc:
            return inp0.like()

        @trtp.aot_impl("example::elemwise_add_plugin")
        def add_plugin_aot_impl(
            inp0: trtp.TensorDesc, block_size: int, outputs: Tuple[trtp.TensorDesc], tactic: int
        ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:

            src = triton.compiler.ASTSource(
                fn=add_kernel,
                signature="*fp32,i32,*fp32",
                constants={
                    "BLOCK_SIZE": block_size,
                },
            )

            compiled_kernel = triton.compile(src)

            launch_params = trtp.KernelLaunchParams()
            N = inp0.shape_expr.numel()
            launch_params.grid_x = trtp.cdiv(N, block_size) 
            launch_params.block_x = compiled_kernel.metadata.num_warps * 32
            launch_params.shared_mem = compiled_kernel.metadata.shared

            extra_args = trtp.SymIntExprs(1)
            extra_args[0] = trtp.SymInt32(N)

            return compiled_kernel.metadata.name, compiled_kernel.asm["ptx"], launch_params, extra_args

        inp = tp.iota((2, 2))
        out = tp.plugin(
            "example::elemwise_add_plugin",
            [inp],
            block_size = 256,
        )

        assert cp.allclose(cp.from_dlpack(out), cp.from_dlpack(inp + 1))
    """


    namespace, name = op.split("::")

    try:
        trtp_op = getattr(getattr(trtp.op, namespace), name)
    except AttributeError:
        raise_error(f"Plugin {op} not found. Expected a plugin of the form '<namespace>::<name>'.")

    attrs = {}
    for key, value in kwargs.items():
        if key not in trtp_op.input_attrs:
            raise_error(
                f"Unexpected attribute {key} provided. Expected one of {trtp_op.input_attrs.keys()}."
            )

        attr_annotation = trtp_op.input_attrs[key]

        if isinstance(value, np.ndarray):
            kwargs[key] = value.tolist()
        else:
            if attr_annotation is not type(value):
                raise_error(
                    f"Unexpected type '{type_str_from_arg(value)}' for attribute '{key}'. Expected '{str_from_type_annotation(attr_annotation)}'."
                )

        attrs[key] = value

    input_descs = [None] * len(inputs)
    for i in range(len(inputs)):
        input_descs[i] = trtp._tensor.TensorDesc()
        input_descs[i].dtype = inputs[i].dtype
        input_descs[i].shape_expr = trtp._tensor.ShapeExprs(inputs[i].rank, _is_dummy=True)
        input_descs[i]._immutable = True
    output_descs = trtp_op.register_func(*input_descs, attrs)
    if not isinstance(output_descs, Tuple):
        output_descs = tuple([output_descs])

    output_info = [(len(desc.shape_expr), desc.dtype) for desc in output_descs]

    return op_utils.create_op(
        Plugin,
        inputs,
        name,
        "1",
        namespace,
        output_info,
        kwargs,
    )
