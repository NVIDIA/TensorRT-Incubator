# Custom Operations

Plugins allow you to extend TensorRT with custom operations.

- The **quickly deployable plugin** (QDP) framework is the easiest way to write plugins.


## Implementing The Plugin

In this guide, we'll implement a plugin that increments a tensor by 1.

:::{seealso}
[TensorRT's guide on QDPs](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/pluginGuide.html)
includes more details on implementing plugins.
:::

We must:

1. **Register the interface** for the plugin.
2. Implement the **plugin kernel**.
3. Generate [**PTX**](https://docs.nvidia.com/cuda/parallel-thread-execution/).


### Registering The Plugin Interface

[`trtp.register`](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/tensorrt.plugin/trt_plugin_register.html#tensorrt.plugin.register)
decorates a function that defines the plugin interface:

```py
import tensorrt.plugin as trtp

# Plugin IDs are of the form: "<namespace>::<name>" and
# uniquely identify a plugin.
INCREMENT_PLUGIN_ID = "example::increment"

@trtp.register(INCREMENT_PLUGIN_ID)
def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> trtp.TensorDesc:
    """
    Defines the plugin interface - inputs, outputs and attributes.

    Args:
        inp0: Input tensor descriptor
        block_size: Block size for the Triton kernel

    Returns:
        Output tensor descriptor with same shape/dtype as input
    """
    return inp0.like()
```

### Implementing The Kernel

For this example, we use [OpenAI's Triton language](https://triton-lang.org/main/index.html)
to implement the kernel:

```py
import triton
import triton.language as tl

@triton.jit # doc: ignore-line
def increment(x_ptr, num_elements, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + 1, mask=mask)
```

<!-- Tripy: DOC: OMIT Start -->
<!-- Hack to make source code inspect work - the decorator tries to inspect the source
    code before we have injected it, so we need to invoke it *after* the function definition -->
```py
# doc: no-print-locals
increment.__globals__.update({"tl": tl}) # This is required to make `tl` available during `triton.compile`.
increment = triton.jit(increment)
```
<!-- Tripy: DOC: OMIT End -->

:::{note}
Kernels can be written in many other ways, e.g. CUDA, CUTLASS, Numba, etc. as long as we can emit PTX.
:::


### Retrieving PTX

[`trtp.aot_impl`](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/tensorrt.plugin/trt_plugin_aot_impl/index.html#tensorrt.plugin.aot_impl)
decorates a function that retrieves PTX, launch parameters, and any extra scalar arguments:

<!-- TODO (pranavm): Remove the _drop_unused_entry_params workaround. -->
```py
from typing import Tuple, Union
import tensorrt.plugin as trtp


@trtp.aot_impl(INCREMENT_PLUGIN_ID)
def increment_aot_impl(
    inp0: trtp.TensorDesc, block_size: int, outputs: Tuple[trtp.TensorDesc], tactic: int
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:

    def _drop_unused_entry_params(ptx: str, kernel_name: str) -> str:
        """
        Removes unreferenced PTX entry parameters for the given kernel.

        NOTE: This is a temporary workaround and will not be necessary in a
            future version of TensorRT.

        Why this exists:
        - Newer Triton versions may add extra kernel entry parameters for
          runtime plumbing.
        - For simple kernels, these extra params can be unreferenced
          in the PTX body.
        - Some plugin launch paths expect only the explicitly
          modeled arguments.

        This helper keeps only parameters that are actually referenced by
        `ld.param ... [<param_name>]` in the PTX body.
        """
        import re

        lines = ptx.splitlines()
        entry_start = next((i for i, line in enumerate(lines) if f".entry {kernel_name}(" in line), None)
        if entry_start is None:
            return ptx

        entry_end = next((i for i in range(entry_start + 1, len(lines)) if lines[i].strip() == ")"), None)
        if entry_end is None:
            return ptx

        param_lines = lines[entry_start + 1 : entry_end]
        body = "\n".join(lines[entry_end + 1 :])

        def param_name(line: str):
            match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*,?\s*$", line)
            return match.group(1) if match and ".param" in line else None

        used = {name for line in param_lines if (name := param_name(line)) and re.search(rf"\[{re.escape(name)}\]", body)}
        filtered_params = [line for line in param_lines if (name := param_name(line)) is None or name in used]
        if len(filtered_params) == len(param_lines):
            return ptx

        for i in range(len(filtered_params) - 1, -1, -1):
            if ".param" in filtered_params[i]:
                filtered_params[i] = filtered_params[i].rstrip().rstrip(",")
                break

        return "\n".join(lines[: entry_start + 1] + filtered_params + lines[entry_end:])

    src = triton.compiler.ASTSource(
        fn=increment,
        signature={"x_ptr": "*fp32", "num_elements": "i32", "y_ptr": "*fp32"},
        constexprs={
            "BLOCK_SIZE": block_size,
        },
    )

    compiled_kernel = triton.compile(src)
    metadata = compiled_kernel.metadata

    # Set the grid, block dims and shared memory for the
    # kernel (as symbolic expressions)
    launch_params = trtp.KernelLaunchParams()
    num_elements = inp0.shape_expr.numel()

    launch_params.grid_x = trtp.cdiv(num_elements, block_size)
    launch_params.block_x = metadata.num_warps * 32
    launch_params.shared_mem = metadata.shared

    # Define extra scalar arguments for the
    # kernel (as symbolic expressions)
    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(num_elements)

    # Optional compatibility step for environments where Triton emits extra
    # unreferenced entry parameters.
    ptx = _drop_unused_entry_params(compiled_kernel.asm["ptx"], metadata.name)

    return metadata.name, ptx, launch_params, extra_args
```


## Using The Plugin

We can use the plugin with {func}`nvtripy.plugin`:

```py
inp = tp.iota((2, 2))
# Plugin attributes are passed as keyword arguments and must match
# the attributes specified by the registration function.
out = tp.plugin(INCREMENT_PLUGIN_ID, [inp], block_size=256)
assert tp.equal(out, inp + 1) # doc: omit
```
