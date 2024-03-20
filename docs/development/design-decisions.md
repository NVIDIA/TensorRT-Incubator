# Design Decisions

This document outlines some of the design decisions we've made and explains why they
were chosen over alternatives.

```{contents} Table of Contents
:depth: 3
```

## No API Redundancy

One way we deviate from frameworks like PyTorch is that our API only provides one way to do things.
For example, in PyTorch, `softmax` is exposed as `torch.softmax`, `torch.nn.functional.softmax`,
`torch.Tensor.softmax`, `torch.nn.Softmax`, etc. All of these are functionally identical and unnecessarily
increase the API surface. In Tripy, we only expose `tripy.softmax`. Trying to use anything else will
result in a (helpful) error message:

```
>>> tp.Tensor([1, 2, 3]).softmax
...
AttributeError: Module: 'tripy.Tensor' does not have attribute: 'softmax'. Did you mean: 'tripy.softmax'?
```


## Why `FlatIR`?

In a previous version of Tripy, we used to map `Trace` operators directly to MLIR
operators. We eventually realized this was creating a lot of code duplication and had two
options:

1. Refactor common code into helper methods that can be shared across `Trace` operators.

2. Introduce another IR between `Trace` and MLIR that would handle the complexity.

(2) is effectively a more organized form of (1). The `FlatIR` is essentially a usability layer
on top of MLIR but has the added benefit that we can inspect and modify the IR without
resorting to the poorly documented (as of this writing), MLIR Python APIs.


## Why Does `to_flat_ir()` Rely On Binding To Input/Output Tensors?

The `to_flat_ir()` method of `Trace` operators generates a subgraph and binds it
to the `inputs` and `outputs` parameters.
For example:

```py
def to_flat_ir(self, inputs, outputs):
    from tripy.flat_ir.ops import TanhOp

    TanhOp.build(inputs, outputs)
```

### Alternative 1: Returning Operators

In a previous version of Tripy, we would return a list of operators that we wanted in the
`FlatIR` subgraph. Our example code would have looked something like this:

```py
def to_flat_ir(self, inputs, outputs):
    from tripy.flat_ir.ops import TanhOp

    tanh = TanhOp.build(inputs, outputs)
    return [tanh]
```

This seems perfectly fine for small cases like this and indeed is the reason that we initially
did it this way. However, for large subgraphs, it quickly becomes error-prone. It is too easy
to forget to include some `FlatIR` operations and that can lead to confusing bugs.

Since you never create a `FlatIR` operation in `to_flat_ir()` that you *don't* want to be
part of the subgraph, we decided to make the `build()` function the single
source of truth for which operations are defined in the subgraph.

### Alternative 2: Functional APIs

Another possibility would have been to provide functional APIs which would let us more easily
build up the FlatIR subgraph. Then our example code could have been implemented like so:

```py
def to_flat_ir(self):
    return tanh(self.inputs[0])
```

This is clearly a nicer interface, but would require us to maintain another set of APIs.
We concluded that this would require too much effort and maintenance overhead.


## Why Is `to_mlir()` Asymmetric With `to_flat_ir()`?

In `to_flat_ir()`, the input and output `FlatIRTensor`s are created by the caller and
the operator that implements `to_flat_ir()` is responsible for creating a `FlatIR` subgraph
that binds to these input and output tensors.

However, in `to_mlir()`, we only pass in the input tensors. The reason for the asymmetry is
that some MLIR operators create their own output tensors whereas others need to bind to
existing ones. Due to this, we leave it to the `FlatIR` operator to convert its own outputs
to MLIR if needed or return the newly created ones.
