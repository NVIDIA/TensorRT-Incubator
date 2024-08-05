[**Installation**](#installation) | [**Quickstart**](#quickstart) | [**Documentation**](#documentation) | [**Examples**](#examples)

# Tripy: A Python Programming Model For TensorRT

Tripy is a Python programming model for [TensorRT](https://developer.nvidia.com/tensorrt) that aims to provide an excellent
user experience without compromising performance. Some of the features of Tripy include:

- **Intuitive API**: Tripy doesn't reinvent the wheel: If you have used NumPy or
    PyTorch before, Tripy APIs should feel familiar.

- **Excellent Error Messages**: When something goes wrong, Tripy tries to provide
    informative and actionable error messages. Even in cases where the error comes
    from deep within the software stack, Tripy is able to map it back to the Python code
    that caused it.


## Installation

```bash
pip3 install git+https://github.com/NVIDIA/TensorRT-Incubator@mlir-tensorrt-v0.1.29#egg=mlir_tensorrt_compiler
pip3 install git+https://github.com/NVIDIA/TensorRT-Incubator@mlir-tensorrt-v0.1.29#egg=mlir_tensorrt_runtime
pip3 install git+https://github.com/NVIDIA/TensorRT-Incubator@tripy-v0.0.1
```

If you want to build from source, please follow the instructions in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Quickstart

In lazy mode evalulation, Tripy defers computation until it is actually needed:

```py
import tripy as tp
import os

def add(a, b):
    return a + b

print(add(tp.Tensor([1., 2.]), tp.Tensor([1.])))
#tensor([2.0000, 3.0000], dtype=float32, loc=gpu:0, shape=(2,))
```

Tripy can compile functions to generate efficient machine code for faster execution:

```py
compiler = tp.Compiler(add)

# a is a 1D dynamic shape tensor of shape (d,), where `d` can range from 1 to 5.
# `[1, 2, 5]` indicates a range from 1 to 5, with optimization for `d = 2`.
a_info = tp.InputInfo(shape=([1, 2, 5],), dtype=tp.float32)

# `b` is a 1D tensor of shape (1,).
b_info = tp.InputInfo((1,), dtype=tp.float32)

compiled_add = compiler.compile(a_info, b_info)

print(compiled_add(tp.Tensor([1., 2., 3.]), tp.Tensor([3.])))
# tensor([4.0000, 5.0000, 6.0000], dtype=float32, loc=gpu:0, shape=(3,))

# Save the compile executable to disk.
executable_file = os.path.join(os.getcwd(), "add_executable.json")
compiled_add.save(executable_file)
```


<!-- TODO (#release): Link to intro to tripy guide -->


<!-- Tripy: DOC: OMIT Start -->

## Documentation

<!-- TODO (#release): Link to docs -->


## Examples

The [examples](./examples/) directory includes end-to-end examples.


## Contributing

For information on how you can contribute to this project, see [CONTRIBUTING.md](./CONTRIBUTING.md)

<!-- Tripy: DOC: OMIT End -->
