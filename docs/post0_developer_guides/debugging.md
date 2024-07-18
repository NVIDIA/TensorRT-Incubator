# Debugging MLIR-TensorRT backend

1. Install new python bindings for compiler and runtime. Assuming `tripy/mlir-tensorrt` directory exists. No need to update `LD_LIBRARY_PATH`.
```bash
python3 -m pip install --force-reinstall mlir-tensorrt/build/wheels/trt100/**/*.whl
```

2. Set environment flags for debugging:

- `export TRIPY_MLIR_DEBUG_ENABLED=1` to enable MLIR-TRT debugging. It will enable debugging prints in MLIR-TRT as well as dump all intermediate IRs after each pass.
- `export TRIPY_MLIR_DEBUG_PATH=<mlir-debug-path>` to set debug path for MLIR-TRT dumps. Default path is `mlir-dumps` under the repo directory. This will create one or more folders named like `module_ins_t1_outs_t2_1`.
- `export TRIPY_TRT_DEBUG_ENABLED=1` to enable TensorRT debugging. It will dump TensorRT engines and their layer information (if there are any TensorRT built during compilation).
- `export TRIPY_TRT_DEBUG_PATH=<trt-debug-path>` to set debug path for TensorRT dumps. Default path is `tensorrt-dumps` under the repo directory.


3. Use LLDB for debugging MLIR-TensorRT backend.
In order to use `lldb` in tripy container, launch the container with extra security options:

```bash
docker run --gpus all --cap-add=SYS_PTRACE \
	--security-opt seccomp=unconfined --security-opt apparmor=unconfined \
	-v $(pwd):/tripy/ -it --rm tripy:latest
```
See https://forums.swift.org/t/debugging-using-lldb/18046 for more details.
