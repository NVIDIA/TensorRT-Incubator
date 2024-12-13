# Using the Compiler

Modules and functions can be compiled ahead of time for better runtime performance.

*Note that the compiler imposes some requirements on the functions/modules it can compile.*
*See {func}`tripy.compile` for details.*

In this guide, we'll work with the [GEGLU](https://arxiv.org/abs/2002.05202v1) module defined below:

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

We can run this in eager mode like usual:

```py
# doc: no-print-locals layer
layer = GEGLU(2, 8)

inp = tp.ones((1, 2))
out = layer(inp)
```

## Compiling

Let's optimize the module using {func}`tripy.compile`.

When we compile in Tripy, we need to provide shape and data type information about each runtime input
using the {class}`tripy.InputInfo` API. Other parameters to the function will be considered compile-time
constants and will be folded into the compiled function.

`GEGLU` only has one input, for which we'll create an `InputInfo` like so:
```py
# doc: no-print-locals
inp_info = tp.InputInfo(shape=(1, 2), dtype=tp.float32)
```

Then we'll compile, which will give us a {class}`tripy.Executable` that we can run:

```py
# doc: no-print-locals fast_geglu
fast_geglu = tp.compile(layer, args=[inp_info])

out = fast_geglu(inp)
```

## Dynamic Shapes

When we compiled above, we used a static shape of `(1, 2)` for the input.
Tripy also supports specifying a range of possible values for each dimension like so:

```py
inp_info = tp.InputInfo(shape=((1, 8, 16), 2), dtype=tp.float32)
```

The shape we used above indicates that the 0th dimension should support a range of values
from `1` to `16`, optimizing for a value of `8`. For the 1st dimension, we continue using
a fixed value of `2`.

Let's compile again with our updated `InputInfo` and try changing the input shape:

```py
# doc: no-print-locals fast_geglu
fast_geglu = tp.compile(layer, args=[inp_info])

# We'll run with the input we created above, which is of shape (1, 2)
out0 = fast_geglu(inp)

# Now let's try an input of shape (2, 2):
inp1 = tp.Tensor([[1., 2.], [2., 3.]], dtype=tp.float32)
out1 = fast_geglu(inp1)
```

If we try using a shape outside of the valid range, the executable will throw a nice error:

<!-- Tripy: TEST: XFAIL Start -->
```py
# doc: allow-exception
inp = tp.ones((32, 2), dtype=tp.float32)
print(fast_geglu(inp))
```
<!-- Tripy: TEST: XFAIL End -->


## Saving The Executable

You can serialize and save executables like so:

```py
# doc: no-print-locals
import tempfile # doc: omit
import os

out_dir = tempfile.mkdtemp() # doc: omit
# Assuming `out_dir` is the directory where you'd like to save the executable:
executable_file_path = os.path.join(out_dir, "executable.json")
fast_geglu.save(executable_file_path)
```

Then you can load and run it again:

```py
# doc: no-print-locals loaded_fast_geglu
loaded_fast_geglu = tp.Executable.load(executable_file_path)

out = loaded_fast_geglu(inp)
os.remove(executable_file_path) # doc: omit
```
