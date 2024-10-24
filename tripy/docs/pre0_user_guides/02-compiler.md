# Using the Compiler



## Model Compilation And Deployment

Let's walk through a simple example of a [GEGLU](https://arxiv.org/abs/2002.05202v1) module defined below:

```py
# doc: no-print-locals
class GEGLU(tp.Module):
    def __init__(self, dim_in, dim_out):
        self.proj = tp.Linear(dim_in, dim_out * 2)
        self.dim_out = dim_out

    def __call__(self, x):
        proj = self.proj(x)
        x, gate = tp.split(proj, 2, proj.rank - 1)
        return x * tp.gelu(gate)
```

To run `GEGLU` in eager mode:

```py
# doc: no-print-locals
layer = GEGLU(2, 8)
inp = tp.ones((1, 2))
out = layer(inp)
```

Now, let's try to optimize this model for inference using Tripy's {func}`tripy.compile`.

When we compile our module, we need to provide information about each input using {class}`tripy.InputInfo`.
The first argument for `InputInfo` is `shape`, where we specify either the static or
dynamic shape information for each dimension. In the example below, we assume the
shape of `inp` is static (`(1, 2)`). The second argument specifies the `dtype` for the input:

```py
# doc: no-print-locals
inp_info = tp.InputInfo(shape=(1, 2), dtype=tp.float32)
```
Now, we can call the `compile` function to obtain a compiled function and use it for inference:

```py
# doc: no-print-locals
fast_geglu = tp.compile(layer, args=[inp_info])
fast_geglu(inp).eval()
```

### Optimization Profiles

In the example above, we assumed `inp` has a static shape of `(1, 2)`.
Now, let's assume that the shape of `inp` can vary from `(1, 2)` to `(16, 2)`, with `(8, 2)`
being the shape we'd like to optimize for. To express this constraint to the compiler,
we can provide the range of shapes to `InputInfo` using `shape=([1, 8, 16], 2)`.
This indicates to the compiler that the first dimension can vary from 1 to 16,
and it should optimize for a size of 8.

```py
# doc: print-locals out out_change_shape
inp_info = tp.InputInfo(shape=([1, 8, 16], 2), dtype=tp.float32)
fast_geglu = tp.compile(layer, args=[inp_info])
out = fast_geglu(inp)

# Let's change the shape of input to (2, 2)
inp = tp.Tensor([[1., 2.], [2., 3.]], dtype=tp.float32)
out_change_shape = fast_geglu(inp)
```

If we provide an input that does not comply with the dynamic shape constraint
given to the compiler, `Tripy` will produce an error with relevant information:

<!-- Tripy: TEST: IGNORE Start -->
```py
# doc: allow-exception
inp = tp.ones((32, 2), dtype=tp.float32)
print(fast_geglu(inp))
```
<!-- Tripy: TEST: IGNORE End -->

### Saving The Executable

A compiled executable can be saved to disk and then used for deployment.

Saving an executable to disk:

```py
# doc: no-print-locals
import tempfile # doc: omit
import os

out_dir = tempfile.mkdtemp() # doc: omit
executable_file_path = os.path.join(out_dir, "executable.json")
fast_geglu.save(executable_file_path)
```

Reading an executable and running inference:

```py
# doc: no-print-locals
loaded_fast_geglu = tp.Executable.load(executable_file_path)

inp = tp.Tensor([[1., 2.], [2., 3.]], dtype=tp.float32)
out = loaded_fast_geglu(inp)
os.remove(executable_file_path) # doc: omit
```

### Querying Executable Properties

We can also query properties about the executable:

```py
# doc: print-locals input_info output_info
input_info = loaded_fast_geglu.get_input_info()
output_info = loaded_fast_geglu.get_output_info()
```
