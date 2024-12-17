
# Tripy: A Python Programming Model For TensorRT

<!-- Tripy: DOC: OMIT Start -->
[**Installation**](#installation) | [**Getting Started**](#getting-started) | [**Documentation**](https://nvidia.github.io/TensorRT-Incubator/) | [**Notebooks**](./notebooks) | [**Examples**](./examples) | [**Contributing**](./CONTRIBUTING.md)

[![Tripy L1](https://github.com/NVIDIA/TensorRT-Incubator/actions/workflows/tripy-l1.yml/badge.svg)](https://github.com/NVIDIA/TensorRT-Incubator/actions/workflows/tripy-l1.yml)
<!-- Tripy: DOC: OMIT End -->

Tripy is a Python programming model for [TensorRT](https://developer.nvidia.com/tensorrt) that aims to provide
an excellent user experience without compromising performance. Some of the goals of Tripy are:

- **Intuitive API**: Tripy doesn't reinvent the wheel: If you have used NumPy or
    PyTorch before, Tripy APIs should feel familiar.

- **Excellent Error Messages**: When something goes wrong, Tripy tries to provide
    informative and actionable error messages. Even in cases where the error comes
    from deep within the software stack, Tripy is usually able to map it back to the
    related Python code.

- **Friendly Documentation**: The documentation is meant to be accessible and comprehensive,
    with plenty of examples to illustrate important points.


## Installation

<!-- Tripy: DOC: OMIT Start -->
### Installing Prebuilt Wheels
<!-- Tripy: DOC: OMIT End -->

```bash
python3 -m pip install nvtripy -f https://nvidia.github.io/TensorRT-Incubator/packages.html
```

<!-- Tripy: DOC: OMIT Start -->
### Building Wheels From Source

To get the latest changes in the repository, you can build Tripy wheels from source.

1. Make sure `build` is installed:

    ```bash
    python3 -m pip install build
    ```

2. From the [`tripy` root directory](.), run:

    ```bash
    python3 -m build . -w
    ```

3. Install the wheel, which should have been created in the `dist/` directory.
    From the [`tripy` root directory](.), run:

    ```bash
    python3 -m pip install -f https://nvidia.github.io/TensorRT-Incubator/packages.html dist/nvtripy-*.whl
    ```

4. **[Optional]** To ensure that Tripy was installed correctly, you can run a sanity check:

    ```bash
    python3 -c "import nvtripy as tp; x = tp.ones((5,), dtype=tp.int32); assert x.tolist() == [1] * 5"
    ```

<!-- Tripy: DOC: OMIT End -->

## Getting Started

We've included several guides in Tripy to make it easy to get started.
We recommend starting with the
[Introduction To Tripy](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/00-introduction-to-tripy.html)
guide.

Other features covered in our guides include:

- [Compiling code (including dynamic shape support)](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/02-compiler.html)
- [Quantization](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/01-quantization.html)

To get an idea of the look and feel of Tripy, let's take a look at a short code example.
All of the features used in this example are explained in more detail in the
introduction guide mentioned above.

```py
# Define our model:
class Model(tp.Module):
    def __init__(self):
        self.conv = tp.Conv(in_channels=1, out_channels=1, kernel_dims=[3, 3])

    def __call__(self, x):
        x = self.conv(x)
        x = tp.relu(x)
        return x


# Initialize the model and populate weights:
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
