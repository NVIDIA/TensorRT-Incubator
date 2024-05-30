# Debugging MLIR-TensorRT backend

1. Install new python bindings for compiler and runtime. Assuming `tripy/mlir-tensorrt` directory exists. No need to update `LD_LIBRARY_PATH`.
```bash
python3 -m pip install --force-reinstall mlir-tensorrt/build/wheels/trt100/**/*.whl
```

2. Update MLIR Debug options in [tripy/config.py](source:/tripy/config.py).
 	`export TRIPY_MLIR_DEBUG_ENABLED=1` to dump IR with below default settings.
```py
# doc: no-eval
import os
# MLIR Debug options
enable_mlir_debug = os.environ.get("TRIPY_MLIR_DEBUG_ENABLED", "0") == "1"
mlir_debug_types = ["-translate-to-tensorrt"]
mlir_debug_tree_path = os.path.join("/", "tripy", "mlir-dumps")
```

3. Use LLDB for debugging MLIR-TensorRT backend.
In order to use `lldb` in tripy container, launch the container with extra security options:

```bash
docker run --gpus all --cap-add=SYS_PTRACE \
	--security-opt seccomp=unconfined --security-opt apparmor=unconfined \
	-v $(pwd):/tripy/ -it --rm tripy:latest
```
See https://forums.swift.org/t/debugging-using-lldb/18046 for more details.
