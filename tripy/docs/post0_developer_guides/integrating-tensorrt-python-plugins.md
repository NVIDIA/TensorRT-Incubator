# Integrating TensorRT Python Plugins with Tripy

TensorRT plugins allow you to extend TensorRT with custom operations. The easiest way to write and integrate a TensorRT plugin with Tripy is to use TensorRT's quickly deployable plugin (QDP) framework.

Let's implement a simple element-wise add plugin as an example.

## Implementation

### Plugin Definition

First, we need to define our plugin using TensorRT's QDP framework. This involves:
1. Registering the plugin with input/output tensor descriptors and plugin attributes
2. Implementing the plugin kernel and providing TensorRT with the corresponding PTX.

```py
import triton
import triton.language as tl
import tensorrt.plugin as trtp
import numpy as np


@triton.jit
def add_kernel(x_ptr, n_elements, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + 1, mask=mask)


@trtp.register("example::elemwise_add_plugin")
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


@trtp.aot_impl("example::elemwise_add_plugin")
def add_plugin_aot_impl(
    inp0: trtp.TensorDesc, block_size: int, outputs: Tuple[trtp.TensorDesc], tactic: int
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    """
    Implements the plugin using a Triton kernel.
    """
    src = triton.compiler.ASTSource(
        fn=add_kernel,
        signature="fp32,i32,fp32",
        constants={
            "BLOCK_SIZE": block_size,
        },
    )

    compiled_kernel = triton.compile(src)
    launch_params = trtp.KernelLaunchParams()

    # Set the grid, block dims and shared memory for the kernel (as symbolic expressions)
    N = inp0.shape_expr.numel()
    launch_params.grid_x = trtp.cdiv(N, block_size)
    launch_params.block_x = compiled_kernel.metadata.num_warps * 32
    launch_params.shared_mem = compiled_kernel.metadata.shared

    # Define extra scalar arguments for the kernel (as symbolic expressions)
    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(N)

    return (
        compiled_kernel.metadata.name,
        compiled_kernel.asm["ptx"],
        launch_params,
        extra_args,
    )
```

### Using the Plugin in Tripy

Once we have defined our plugin, it is accessible under `trt.plugin.op.<namespace>.<plugin_name>`, which is `trt.plugin.op.example.elemwise_add_plugin` for the above plugin. `tp.plugin` can be used to add the plugin to Tripy.

```py
inp = tp.iota((2, 2))
out = tp.plugin(
    trt.plugin.op.example.elemwise_add_plugin,
    [inp],  
    block_size = 256,
)

assert cp.allclose(cp.from_dlpack(out), cp.from_dlpack(inp + 1))
```

### Supported attribute types

QDPs support either scalar attributes of selected primitive Python types or 1-D Numpy arrays of selected Numpy data types. For Numpy array attributes, the corresponding argument in `@trtp.register` must be annotated with `numpy.typing.NDArray[dtype]`, where `dtype` is the expected Numpy data type. Please refer to the [TensorRT Plugin Guide](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/pluginGuide.html) for details on exact primitive and Numpy data types supported by QDPs.

## Understanding the Implementation

### Plugin Registration

The `@trtp.register` decorator registers our plugin with TensorRT. The plugin name follows the format 
`namespace::plugin_name`. The registration function:

1. Takes input tensor descriptors and plugin attributes as arguments
2. Returns output tensor descriptors
3. Defines the plugin interface that Tripy will use

### Plugin Implementation 

The `@trtp.aot_impl` decorator returns an ahead-of-time (AOT) compiled kernel, along with information required to invoke the kernel such as kernel launch parameters and extra scalar arguments. This plugin uses OpenAI Triton to define the kernel and uses its AOT compilation capabilities to retrieve PTX for the kernel.

### Tripy Integration

Tripy's `tp.plugin()` function accepts:
- The plugin op
- A list of input tensors
- Plugin attributes as keyword arguments

Links:
- [TensorRT Plugin Guide](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/pluginGuide.html)
- [Triton Documentation](https://triton-lang.org/main/index.html)
