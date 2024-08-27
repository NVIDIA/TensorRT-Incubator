<!-- Tripy: DOC: OMIT Start -->
[**Installation**](#installation) | [**Quickstart**](#quickstart) | [**Documentation**](#documentation) | [**Examples**](#examples)
<!-- Tripy: DOC: OMIT End -->

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
python3 -m pip install --no-index -f https://nvidia.github.io/TensorRT-Incubator/packages.html tripy --no-deps
python3 -m pip install -f https://nvidia.github.io/TensorRT-Incubator/packages.html tripy
```

<!-- Tripy: DOC: OMIT Start -->
If you want to build from source, please follow the instructions in [CONTRIBUTING.md](./CONTRIBUTING.md).
<!-- Tripy: DOC: OMIT End -->

## Quickstart

In eager mode, Tripy works just like you'd expect:

```py
# doc: no-print-locals
import tripy as tp

a = tp.Tensor([1.0, 2.0])
print(a + 1)
```

Tripy can also compile functions to generate efficient machine code for faster execution:

```py
# doc: no-print-locals
def add(a, b):
    return a + b

compiler = tp.Compiler(add)

# When compiling, we need to specify shape and data type constraints on the inputs:

# a is a 1D dynamic shape tensor of shape (d,), where `d` can range from 1 to 5.
# `[1, 2, 5]` indicates a range from 1 to 5, with optimization for `d = 2`.
a_info = tp.InputInfo(shape=([1, 2, 5],), dtype=tp.float32)

# `b` is a 1D tensor of shape (1,).
b_info = tp.InputInfo((1,), dtype=tp.float32)

compiled_add = compiler.compile(a_info, b_info)

print(compiled_add(tp.Tensor([1., 2., 3.]), tp.Tensor([3.])))
```

For more details, see the
[Introduction To Tripy](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/00-introduction-to-tripy.html)
guide.


<!-- Tripy: DOC: OMIT Start -->

## Documentation

The documentation is hosted [here](https://nvidia.github.io/TensorRT-Incubator/).


## Examples

The [examples](./examples/) directory includes end-to-end examples.


## Contributing

For information on how you can contribute to this project, see [CONTRIBUTING.md](./CONTRIBUTING.md)

<!-- Tripy: DOC: OMIT End -->
