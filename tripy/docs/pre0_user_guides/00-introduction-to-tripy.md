# An Introduction To Tripy

## What Is Tripy?

Tripy is a compiler that compiles deep learning models for inference using TensorRT as a backend.
It aims to be fast, easy to debug, and provide an easy-to-use Pythonic interface.

## Your First Tripy Program

But enough talk; let's see some code:

```py
# doc: no-print-locals
a = tp.arange(5)
c = a + 1.5
print(c)
assert np.array_equal(cp.from_dlpack(c).get(), np.arange(5, dtype=np.float32) + 1.5) # doc: omit
```

This should look familiar if you've used linear algebra or deep learning libraries like
NumPy and PyTorch.


### Lazy Evaluation: Putting Off Work

One important point is that Tripy uses a lazy evaluation model; that is,
no computation is performed until a value is actually needed.

In the example above, that means that `c` will not be evaluated until it is used,
such as when we print its values.

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

print(f"Time to create 'c': {end - start:.3f} seconds.")
```

It looks like Tripy is very fast! While Tripy *execution* is very fast, compiling the program
takes some time. The reason the time is so low relative to what we'd expect for initializing
and running the compiler is that *we're not doing that yet*.

The actual compilation and computation only happens when we evaluate `c`:

```py
# doc: no-print-locals
start = time.time()
print(c)
end = time.time()

print(f"Time to print 'c': {end - start:.3f} seconds.")
```

That is why the time to print `c` is so much higher than the time to create it.

If we wanted to time individual parts of the model, we would insert calls to `.eval()`;
for example, adding a `c.eval()` prior to checking the end time would tell us how
long it took to compile and run the subgraph that computes `c`.


## Organizing Code Using Modules

The {class}`tripy.Module` API allows you to create reusable blocks that can be composed together
to create models. Modules may be comprised of other modules, including modules predefined
by Tripy, like {class}`tripy.Linear` and {class}`tripy.LayerNorm`.

For example, we can define a Transfomer MLP block like so:

```py
class MLP(tp.Module):
    def __init__(self, embedding_size, dtype=tp.float32):
        super().__init__()
        self.c_fc = tp.Linear(embedding_size, 4 * embedding_size, bias=True, dtype=dtype)
        self.c_proj = tp.Linear(4 * embedding_size, embedding_size, bias=True, dtype=dtype)

    def __call__(self, x):
        x = self.c_fc(x)
        x = tp.gelu(x)
        x = self.c_proj(x)
        return x
```

To use it, we just need to construct and call it:

```py
mlp = MLP(embedding_size=2)

inp = tp.iota(shape=(1, 2), dim=1, dtype=tp.float32)
out = mlp(inp)
```


## To `compile` Or Not To `compile`

All the code we've seen so far has been using Tripy's eager mode. It is also possible to compile
functions or modules ahead of time, which can result in significantly better performance.

*Note that the compiler imposes some requirements on the functions/modules it can compile.*
*See {class}`tripy.Compiler` for details.*

Let's compile the MLP module we defined above as an example:

```py
# doc: no-print-locals
compiler = tp.Compiler(mlp)

# When we compile, we need to indicate which parameters to the function should be runtime inputs.
# In this case, MLP takes a single input tensor for which we can specify our desired shape and datatype.
fast_mlp = compiler.compile(tp.InputInfo(shape=(1, 2), dtype=tp.float32))
```

It is also possible to compile for a range of possible input shapes.
See {func}`tripy.Compiler.compile` for details.

Now let's benchmark the compiled version against eager mode:
```py
# doc: no-print-locals
start = time.time()
out = mlp(inp)
out.eval() # Recall that we need to evaluate in order to actually materialize `out`
end = time.time()

eager_time = (end - start)
print(f"Eager mode time: {eager_time:.4f} seconds")

ITERS = 10
start = time.time()
for _ in range(ITERS):
    out = fast_mlp(inp)
    out.eval()
end = time.time()

compiled_time = (end - start) / ITERS
print(f"Compiled mode average time: {(end - start) / ITERS:.4f} seconds")
# Make sure compiled mode is actually faster # doc: omit
assert compiled_time < 0.01 * eager_time # doc: omit
```

As you can see, the compiled module is significantly faster than running the module
in eager mode.
