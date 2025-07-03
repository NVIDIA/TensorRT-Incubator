# RUN: %PYTHON %s 2>&1

import mlir_tensorrt.runtime.api as runtime


# Test with multiple types
debug_types = ["allocator", "runtime"]
runtime.GlobalDebug.set_types(debug_types)
runtime.GlobalDebug.flag = True
assert runtime.GlobalDebug.flag == True, "expected global debug flag to be true"
runtime.GlobalDebug.flag = False
assert runtime.GlobalDebug.flag == False, "expected global debug flag to be false"
