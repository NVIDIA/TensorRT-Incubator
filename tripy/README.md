
# Tripy: A Python Programming Model For TensorRT

[**Quick Start**](#quick-start)
| [**Installation**](#installation)
| [**Examples**](https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy/examples)
| [**Notebooks**](https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy/notebooks)
| [**Contributing**](https://github.com/NVIDIA/TensorRT-Incubator/blob/main/tripy/CONTRIBUTING.md)
| [**Documentation**](https://nvidia.github.io/TensorRT-Incubator/)

<!-- Tripy: DOC: OMIT Start -->
[![Tripy L1](https://github.com/NVIDIA/TensorRT-Incubator/actions/workflows/tripy-l1.yml/badge.svg)](https://github.com/NVIDIA/TensorRT-Incubator/actions/workflows/tripy-l1.yml)
<!-- Tripy: DOC: OMIT End -->

**Tripy** is a debuggable, Pythonic frontend for [TensorRT](https://developer.nvidia.com/tensorrt),
a deep learning inference compiler.

What you can expect:

- **High performance** by leveraging [TensorRT](https://developer.nvidia.com/tensorrt)'s optimization capabilties.
- An **intuitive API** that follows conventions of the ecosystem.
- **Debuggability** with features like **eager mode** to interactively debug mistakes.
- **Excellent error messages** that are informative and actionable.
- **Friendly documentation** that is comprehensive but concise, with code examples.


## Installation

```bash
python3 -m pip install nvtripy -f https://nvidia.github.io/TensorRT-Incubator/packages.html
```


## Quick Start

See the
[Introduction To Tripy](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/00-introduction-to-tripy.html)
guide for details:

<!-- Tripy: DOC: NO_PRINT_LOCALS Start -->
- **Defining** a model:

    ```py
    class Model(tp.Module):
        def __init__(self):
            self.conv = tp.Conv(in_channels=1, out_channels=1, kernel_dims=[3, 3])

        def forward(self, x):
            x = self.conv(x)
            x = tp.relu(x)
            return x
    ```

- **Initializing** it:

    ```py
    model = Model()
    model.load_state_dict(
        {
            "conv.weight": tp.ones((1, 1, 3, 3)),
            "conv.bias": tp.ones((1,)),
        }
    )

    dummy_input = tp.ones((1, 1, 4, 4))
    ```

- Executing in **eager mode**:

    ```py
    eager_out = model(dummy_input)
    ```

- **Compiling** and executing:

    ```py
    compiled_model = tp.compile(
        model,
        args=[tp.InputInfo(shape=(1, 1, 4, 4), dtype=tp.float32)],
    )

    compiled_out = compiled_model(dummy_input)
    ```
<!-- Tripy: DOC: NO_PRINT_LOCALS End -->


<!-- Tripy: DOC: OMIT Start -->
## Building Wheels From Source

For the latest changes, build Tripy wheels from source:

1. Install `build`:

    ```bash
    python3 -m pip install build
    ```

2. Build a wheel from the [`tripy` root directory](.):

    ```bash
    python3 -m build . -w
    ```

3. Install the wheel from the [`tripy` root directory](.):

    ```bash
    python3 -m pip install -f https://nvidia.github.io/TensorRT-Incubator/packages.html dist/nvtripy-*.whl
    ```

4. **[Optional]** Sanity check:

    ```bash
    python3 -c "import nvtripy as tp; x = tp.ones((5,), dtype=tp.int32); assert x.tolist() == [1] * 5"
    ```
<!-- Tripy: DOC: OMIT End -->
