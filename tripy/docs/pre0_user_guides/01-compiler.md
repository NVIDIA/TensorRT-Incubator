# Using the Compiler

Modules and functions can be compiled for better performance.

:::{important}
There are **restrictions** on what can be compiled - see {func}`nvtripy.compile`.
:::

We'll demonstrate using a [GEGLU](https://arxiv.org/abs/2002.05202v1) module:

```py
# doc: no-print-locals
class GEGLU(tp.Module):
    def __init__(self, in_dim, out_dim):
        self.proj = tp.Linear(in_dim, out_dim * 2)
        self.out_dim = out_dim

    def forward(self, x):
        proj = self.proj(x)
        x, gate = tp.split(proj, 2, proj.rank - 1)
        return x * tp.gelu(gate)


layer = GEGLU(in_dim=2, out_dim=1)

layer.load_state_dict({
    "proj.weight": tp.ones((2, 2)),
    "proj.bias": tp.ones((2,))
})
```

## Compiling

We must inform the compiler which parameters are **runtime inputs**
and provide their shape/datatypes using {class}`nvtripy.InputInfo`:

```py
# doc: no-print-locals
# GEGLU has one parameter, which needs to be a runtime input:
inp_info = tp.InputInfo(shape=(1, 2), dtype=tp.float32)
fast_geglu = tp.compile(layer, args=[inp_info])
```

:::{note}
Other parameters become **compile-time constants** and will be folded away.
:::

The compiler returns an {class}`nvtripy.Executable`, which behaves like a callable:

```py
inp = tp.ones((1, 2)).eval()
out = fast_geglu(inp)
```

## Dynamic Shapes

To enable dynamic shapes, we can specify a range for any given dimension:

```py
inp_info = tp.InputInfo(shape=((1, 2, 4), 2), dtype=tp.float32)
```

`((1, 2, 4), 2)` means:

- The **0th** dimension should support sizes from `1` to `4`, optimizing for `2`.
- The **1st** dimension should support a fixed size of `2`.

The executable will support inputs within this range of shapes:
```py
# doc: no-print-locals fast_geglu
fast_geglu = tp.compile(layer, args=[inp_info])

# Use the input created previously, of shape: (1, 2)
out0 = fast_geglu(inp)

# Now use an input with a different shape: (2, 2):
inp1 = tp.Tensor([[1., 2.], [2., 3.]]).eval()
out1 = fast_geglu(inp1)
```

### Named Dynamic Dimensions

Dynamic dimensions can be **named** using {class}`nvtripy.NamedDimension`.

- Dimensions with the same name must be **equal** at runtime.
- The compiler can exploit this equality to make **better optimizations**.

```py
# doc: no-print-locals fast_add
def add(a, b):
    return a + b

batch = tp.NamedDimension("batch", 1, 2, 4)

# The batch dimension is dynamic but is always equal at runtime for both inputs:
inp_info0 = tp.InputInfo(shape=(batch, 2), dtype=tp.float32)
inp_info1 = tp.InputInfo(shape=(batch, 2), dtype=tp.float32)

fast_add = tp.compile(add, args=[inp_info0, inp_info1])
```


## Saving And Loading Executables

- **Serialize** and **save**:

    ```py
    # doc: no-print-locals
    import tempfile # doc: omit
    import os

    out_dir = tempfile.mkdtemp() # doc: omit
    executable_file_path = os.path.join(out_dir, "executable.json")
    fast_geglu.save(executable_file_path)
    ```

- **Load** and **run**:

    ```py
    # doc: no-print-locals loaded_fast_geglu
    loaded_fast_geglu = tp.Executable.load(executable_file_path)

    out = loaded_fast_geglu(inp)
    os.remove(executable_file_path) # doc: omit
    ```
