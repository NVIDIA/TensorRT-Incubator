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

Due to how our package is currently set up, installation is more complicated than a simple `pip install`.
This should change in the near future, but for now, you can use Tripy by building the development container
as specified in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Quickstart

In eager mode, Tripy works just like you'd expect:

```py
import tripy as tp

a = tp.Tensor([1.0, 2.0])
print(a + 1)

# tensor([2.0000, 3.0000], dtype=float32, loc=gpu:0, shape=(2,))
```


```python
>>> a
tensor([1.0000, 2.0000], dtype=float32, loc=gpu:0, shape=(2,))
```


Output:
```
tensor([2.0000, 3.0000], dtype=float32, loc=gpu:0, shape=(2,))
```


Tripy can also compile functions to generate efficient machine code for faster execution:

```py
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
# tensor([4.0000, 5.0000, 6.0000], dtype=float32, loc=gpu:0, shape=(3,))
```


```python
>>> compiler
<tripy.backend.compiler_api.Compiler object at 0x7f723ca3e770>
>>> a_info
InputInfo(min=(1,), opt=(2,), max=(5,), dtype=float32)
>>> b_info
InputInfo(min=(1,), opt=(1,), max=(1,), dtype=float32)
>>> compiled_add
<tripy.backend.compiler_api.Executable object at 0x7f723c949750>
```


Output:
```
tensor([4.0000, 5.0000, 6.0000], dtype=float32, loc=gpu:0, shape=(3,))
```


For more details, see the
[Introduction To Tripy](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/introduction-to-tripy.html)
guide.

