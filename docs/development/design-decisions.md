# Design Decisions

This document outlines some of the design decisions we've made and explains why they
were chosen over alternatives.

```{contents} Table of Contents
:depth: 3
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


## Why Does `to_flat_ir()` Rely On Constructor Side-effects?

The `to_flat_ir()` method of `Trace` operators currently relies on the side effects of
the `BaseFlatIROp` constructor to build a subgraph. For example:

```py
def to_flat_ir(self, inputs, outputs):
    from tripy.flat_ir.ops import TanhOp

    TanhOp(self, inputs, outputs)
```

Here, `TanhOp` will bind itself to the inputs and outputs on construction.
Below we explain some of the alternatives that were considered and why they were rejected.

### Alternative 1: Manual Everything

The most obvious alternative is to manually set up everything in the subgraph.
Our example code becomes:

```py
def to_flat_ir(self, inputs, outputs):
    from tripy.flat_ir.ops import TanhOp

    tanh = TanhOp(self, inputs, outputs)

    for out in outputs:
        out.producer = tanh
```

Clearly this would create way too much repeated code everywhere; one way to fix that
would be to set the producer(s) of the output(s) in the constructor of the `FlatIR` operator,
which is exactly what we do.

### Alternative 2: Returning Operators

In a previous version of Tripy, we would return a list of operators that we wanted in the
`FlatIR` subgraph. Our example code would have looked something like this:

```py
def to_flat_ir(self, inputs, outputs):
    from tripy.flat_ir.ops import TanhOp

    tanh = TanhOp(self, inputs, outputs)
    return [tanh]
```

This seems perfectly fine for small cases like this and indeed is the reason that we initially
did it this way. However, for large subgraphs, it quickly becomes error-prone. It is too easy
to forget to include some `FlatIR` operations and that can lead to confusing bugs.

Since you never construct a `FlatIR` operation in `to_flat_ir()` that you *don't* want to be
part of the subgraph, we decided to make the constructor effectively behave like the single
source of truth for defined operations in the subgraph.

#### Alternative 3: Functional APIs

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
