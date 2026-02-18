# Force initializtion of tvm_ffi before loading the extension module.
try:
    import tvm_ffi
except Exception as e:
    raise RuntimeError(
        "Error importing tvm_ffi, which is required for mlir-tensorrt-runtime: %s",
        e,
    )

from ._mlir_libs._api import *
