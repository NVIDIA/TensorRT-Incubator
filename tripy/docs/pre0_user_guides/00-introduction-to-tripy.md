# An Introduction To Tripy

## What Is Tripy?

Tripy is a compiler that compiles deep learning models for inference using TensorRT as a backend.
It aims to be fast, easy to debug, and provide an easy-to-use Pythonic interface.

## Your First Tripy Program

```py
# doc: no-print-locals
a = tp.arange(5)
c = a + 1.5
print(c)
assert cp.array_equal(cp.from_dlpack(c), cp.arange(5, dtype=np.float32) + 1.5) # doc: omit
```

This should look familiar if you've used linear algebra or deep learning libraries like
NumPy and PyTorch. Hopefully, the code above is self-explanatory, so we won't go into details.

## Organizing Code Using Modules

The {class}`nvtripy.Module` API allows you to create reusable blocks that can be composed together
to create models. Modules may be comprised of other modules, including modules predefined
by Tripy, like {class}`nvtripy.Linear` and {class}`nvtripy.LayerNorm`.

For example, we can define a Transfomer MLP block like so:

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

To use it, we just need to construct and call it:

```py
# doc: no-print-locals mlp
mlp = MLP(embd_size=2)

inp = tp.iota(shape=(1, 2), dim=1, dtype=tp.float32)
out = mlp(inp)
```

## Compiling Code

All the code we've seen so far has been using Tripy's eager mode. It is also possible to compile
functions or modules ahead of time, which can result in significantly better performance.

*Note that the compiler imposes some requirements on the functions/modules it can compile.*
*See {func}`nvtripy.compile` for details.*

Let's compile the MLP module we defined above as an example:

```py
# doc: no-print-locals
# When we compile, we need to indicate which parameters to the function
# should be runtime inputs. In this case, MLP takes a single input tensor
# for which we can specify our desired shape and datatype.
fast_mlp = tp.compile(mlp, args=[tp.InputInfo(shape=(1, 2), dtype=tp.float32)])
```

Now let's benchmark the compiled version against eager mode:
<!--
```py
from nvtripy.frontend.cache import global_cache

global_cache._cache.clear()
``
-->

```py
# doc: no-print-locals
import time

start = time.time()
out = mlp(inp)
# We need to evaluate in order to actually materialize `out`.
# See the section on lazy evaluation below for details.
out.eval()
end = time.time()

eager_time = (end - start) * 1000
print(f"Eager mode time: {eager_time:.4f} ms")

start = time.time()
out = fast_mlp(inp)
out.eval()
end = time.time()

compiled_time = (end - start) * 1000
print(f"Compiled mode time: {compiled_time:.4f} ms")
# Make sure compiled mode is actually faster # doc: omit
assert compiled_time < 0.01 * eager_time # doc: omit
```

For more information on the compiler, compiled functions/modules, and dynamic shapes,
see the [compiler guide](project:./02-compiler.md).

## Things To Note

### Eager Mode: How Does It Work?

If you've used TensorRT before, you may know that it does not support an eager mode.
In order to provide eager mode support in Tripy, we actually need to compile the graph
under the hood.

Although we employ several tricks to make compile times faster when using eager mode,
we do still need to compile, and so eager mode will likely be slower than other
comparable frameworks.

Consequently, we suggest that you use eager mode primarily for debugging and
compiled mode for deployments.

### Lazy Evaluation: Putting Off Work

One important point is that Tripy uses a lazy evaluation model; that is,
no computation is performed until a value is actually needed.

In most cases, this is simply an implementation detail that you will not notice.
One exception to this is when attempting to time code. Consider the following code:

```py
# doc: no-print-locals
import time

start = time.time()
a = tp.arange(5)
b = tp.arange(5)
c = a + b + tp.tanh(a)
end = time.time()

print(f"Time to create 'c': {(end - start) * 1000:.3f} ms.")
```

Given what we said above about eager mode, it seems like Tripy is very fast!
Of course, this is because *we haven't actually done anything yet*.
The actual compilation and execution only happens when we evaluate `c`:

```py
# doc: no-print-locals
start = time.time()
print(c)
end = time.time()

print(f"Time to print 'c': {(end - start) * 1000:.3f} ms.")
```

That is why the time to print `c` is so much higher than the time to create it.
