# An Introduction To Tripy

**Tripy** is a debuggable, Pythonic frontend for [TensorRT](https://developer.nvidia.com/tensorrt),
a deep learning inference compiler:

```py
# doc: no-print-locals
a = tp.ones((2, 3))
b = tp.ones((2, 3))
c = a + b
print(c)
assert tp.equal(c, tp.full((2, 3), 2.0)) # doc: omit
```

## Organizing Code With Modules

{class}`nvtripy.Module`s are composable, reusable blocks of code:

```py
class MLP(tp.Module):
    def __init__(self, embd_size, dtype=tp.float32):
        super().__init__()
        self.c_fc = tp.Linear(embd_size, 4 * embd_size, bias=True, dtype=dtype)
        self.c_proj = tp.Linear(4 * embd_size, embd_size, bias=True, dtype=dtype)

    def __call__(self, x):
        x = self.c_fc(x)
        x = tp.gelu(x)
        x = self.c_proj(x)
        return x
```

Usage:

```py
# doc: no-print-locals mlp
mlp = MLP(embd_size=2)

inp = tp.iota(shape=(1, 2), dim=1, dtype=tp.float32)
out = mlp(inp)
```

## Compiling For Better Performance

Modules and functions can be **compiled**:

```py
# doc: no-print-locals
fast_mlp = tp.compile(
    mlp,
    # We must indicate which parameters are runtime inputs.
    # MLP takes 1 input tensor for which we specify shape and datatype:
    args=[tp.InputInfo(shape=(1, 2), dtype=tp.float32)],
)
```

Usage:
```py
out = fast_mlp(inp)
```

:::{note}
There are **restrictions** on what can be compiled - see {func}`nvtripy.compile`.
:::

:::{seealso}
The [compiler guide](project:./02-compiler.md) contains more information, including how to enable **dynamic shapes**.
:::


## Pitfalls And Best Practices

- **Best Practice:** Use **eager mode** only for **debugging**; compile for deployment.

    **Why:** Eager mode internally compiles the graph (slow!) as TensorRT lacks eager execution.

- **Pitfall:** Be careful timing code in **eager mode**.

    **Why:** Tensors are evaluated only when used; naive timing will be inaccurate:

    ```py
    # doc: no-print-locals
    import time

    start = time.time()
    a = tp.gelu(tp.ones((2, 8)))
    end = time.time()

    # `a` has not been evaluated yet - this time is not what we want!
    define_time = end - start # doc: omit
    print(f"Defined `a` in: {(end - start) * 1000:.3f} ms.")

    start = time.time()
    # `a` is used (and thus evaluated) for the first time:
    print(a)
    end = time.time()

    # This includes compilation time, not just execution time!
    print(f"Compiled and evaluated `a` in: {(end - start) * 1000:.3f} ms.")
    assert end - start > define_time # doc: omit
    ```
