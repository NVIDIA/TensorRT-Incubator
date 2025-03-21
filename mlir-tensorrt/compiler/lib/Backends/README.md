# MLIR TensorRT Backends

This directory contains backend definitions that define how operations in the input IR
may be grouped and compiled.

In particular, each "backend" defines an attribute which implements the Plan
dialect's `ClusterKindAttrInterface` interface. It implements a set of method that define:

1. What kinds of operations from the input IR can or should be implemented 
   using this particular backend strategy. In other words, what operations
   are "clusterable" for this backend.

2. Once operations are clustered, how should the backend close and outline
   the cluster? Different backends may require different logic here.


## Host Backend

The [Host backend](./Host/) defines clusters of operations that can be offloaded to CPU
execution, either through our Lua interpreter or by targeting LLVM. It handles operations 
that are better suited for CPU execution or that don't have GPU implementations. Examples
might include operations which perform "shape tensor" calculations or "reshape-like" operations
whose runtime implementation amounts to modification of metadata which lives in host memory.


## TensorRT Backend 

The [TensorRT backend](./TensorRT/) defines groups of operations which can be
offloaded to TensorRT.
