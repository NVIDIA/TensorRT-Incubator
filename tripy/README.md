
# Tripy: A Python Programming Model For TensorRT

[**Installation**](#installation)
| [**Getting Started**](#getting-started)
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

- **High Performance:** Leveraging [TensorRT](https://developer.nvidia.com/tensorrt)'s optimization capabilties.

- **Intuitive API:** A familiar API that follows conventions of the ecosystem.

- **Debuggability:** **Eager mode** to interactively debug mistakes.

- **Excellent Error Messages**: Informative and actionable.

- **Friendly Documentation**: Comprehensive but straight-to-the-point, with code examples.

Code is worth 1,000 words:

```py
# doc: no-print-locals
# Define our model:
class Model(tp.Module):
    def __init__(self):
        self.conv = tp.Conv(in_channels=1, out_channels=1, kernel_dims=[3, 3])

    def __call__(self, x):
        x = self.conv(x)
        x = tp.relu(x)
        return x


# Initialize the model and load weights:
model = Model()
model.load_state_dict(
    {
        "conv.weight": tp.ones((1, 1, 3, 3)),
        "conv.bias": tp.ones((1,)),
    }
)

inp = tp.ones((1, 1, 4, 4))

# Eager mode:
eager_out = model(inp)

# Compiled mode:
compiled_model = tp.compile(
    model,
    args=[tp.InputInfo(shape=(1, 1, 4, 4), dtype=tp.float32)],
)

compiled_out = compiled_model(inp)
```


## Installation

<!-- Tripy: DOC: OMIT Start -->
### Installing Prebuilt Wheels
<!-- Tripy: DOC: OMIT End -->

```bash
python3 -m pip install nvtripy -f https://nvidia.github.io/TensorRT-Incubator/packages.html
```

<!-- Tripy: DOC: OMIT Start -->
### Building Wheels From Source

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


## Getting Started

- **Start with**:
    [Introduction To Tripy](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/00-introduction-to-tripy.html)

Other guides:

- [Compiling For Better Performance](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/02-compiler.html)
- [Quantization](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/01-quantization.html)
