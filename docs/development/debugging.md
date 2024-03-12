# Debugging MLIR-TensorRT backend

1. Install new python bindings for mlir-tensorrt-compiler. Assuming `tripy/mlir-tensorrt` directory exists.
```bash
python3 -m pip install mlir-tensorrt/build/wheels/trt100/mlir-tensorrt-compiler-wheel/mlir_tensorrt_compiler-0.1.6+cuda12.trt100-cp310
-cp310-linux_x86_64.whl
```

2. Update LD_LIBRARY_PATH.
```bash
export LD_LIBRARY_PATH=/tripy/mlir-tensorrt/build/lib/Integrations/PJRT:/tripy/mlir-tensorrt/build/python_packages/mlir_tensorrt_runtime/mlir_tensorrt/runtime/_mlir_libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/tripy/mlir-tensorrt/build/python_packages/mlir_tensorrt_compiler/mlir_tensorrt/compiler/_mlir_libs/:$LD_LIBRARY_PATH
```

3. Update MLIR Debug options in [tripy/config.py](source:/tripy/config.py).
 	`export TRIPY_MLIR_DEBUG_ENABLED=1` to dump IR with below default settings.
```py
import os
# MLIR Debug options
MLIR_DEBUG_ENABLED = os.environ.get("TRIPY_MLIR_DEBUG_ENABLED", "0") == "1"
MLIR_DEBUG_TYPES = ["-mlir-print-ir-after-all"]
MLIR_DEBUG_TREE_PATH = os.path.join("/", "tripy", "mlir-dumps")
```

4. Use LLDB for debugging MLIR-TensorRT backend.
In order to use `lldb` in tripy container, launch the container with extra security options:

```bash
docker run --gpus all --cap-add=SYS_PTRACE \
	--security-opt seccomp=unconfined --security-opt apparmor=unconfined \
	-v $(pwd):/tripy/ -it --rm tripy:latest
```
See https://forums.swift.org/t/debugging-using-lldb/18046 for more details.
