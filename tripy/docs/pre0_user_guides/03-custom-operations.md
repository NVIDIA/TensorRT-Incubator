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
decorates a function that retrieves PTX, launch parameters, and any extra arguments:

```py
from typing import Tuple, Union
import tensorrt.plugin as trtp

@trtp.aot_impl(INCREMENT_PLUGIN_ID)
def increment_aot_impl(
    inp0: trtp.TensorDesc, block_size: int, outputs: Tuple[trtp.TensorDesc], tactic: int
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    src = triton.compiler.ASTSource(
        fn=increment,
        signature="*fp32,i32,*fp32",
        constants={
            "BLOCK_SIZE": block_size,
        },
    )

    compiled_kernel = triton.compile(src)

    # Set the grid, block dims and shared memory for the
    # kernel (as symbolic expressions)
    launch_params = trtp.KernelLaunchParams()
    num_elements = inp0.shape_expr.numel()
    launch_params.grid_x = trtp.cdiv(num_elements, block_size)
    launch_params.block_x = compiled_kernel.metadata.num_warps * 32
    launch_params.shared_mem = compiled_kernel.metadata.shared

    # Define extra scalar arguments for the
    # kernel (as symbolic expressions)
    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(num_elements)

    return compiled_kernel.metadata.name, compiled_kernel.asm["ptx"], launch_params, extra_args
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
