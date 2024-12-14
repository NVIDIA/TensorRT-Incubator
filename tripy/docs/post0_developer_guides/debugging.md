# Debugging MLIR-TensorRT

While developing Tripy features, you may need to debug MLIR-TRT code.
This guide outlines some methods of doing so.


## Environment Variables

We include some environment variables to enable extra debugging information from MLIR-TRT:

- `export TRIPY_EAGER_CACHE=1` will enable eager caching for Tripy tensors to cache all intermediate IRs for future reuse. 
- `export TRIPY_MLIR_DEBUG_ENABLED=1` will enable debug prints in MLIR-TRT and dump all intermediate IRs to a directory.
- `export TRIPY_MLIR_DEBUG_PATH=<mlir-debug-path>` sets the directory for IR dumps. The default path is `mlir-dumps`.
- `export TRIPY_TRT_DEBUG_ENABLED=1` will dump TensorRT engines and their layer information.
- `export TRIPY_TRT_DEBUG_PATH=<trt-debug-path>` sets the directory for TensorRT dumps. Default path is `tensorrt-dumps`.


## Using A Debugger

For more involved bugs, it may be helpful to step into MLIR-TRT code.
To do so, you will need a debug build of MLIR-TRT;
see [CONTRIBUTING.md](source:/CONTRIBUTING.md)
for details on using custom builds of MLIR-TRT.

Once you've installed the debug build in the container, you should be able to use `gdb` as normal.

Alternatively, you can use [LLDB](https://lldb.llvm.org/) if you launch the container with extra security options:

<!-- Tripy: DOC: NO_EVAL Start -->
```bash
docker run --gpus all --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -p 8080:8080 -v $(pwd):/tripy/ -it --rm tripy:latest
```
<!-- Tripy: DOC: NO_EVAL End -->

See [this post](https://forums.swift.org/t/debugging-using-lldb/18046) for details on
why these security options are required.
