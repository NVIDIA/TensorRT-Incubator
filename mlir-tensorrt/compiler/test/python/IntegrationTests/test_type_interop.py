# RUN: %PYTHON %s 2>&1 | FileCheck %s

import mlir_tensorrt.compiler.api as compiler

# This is incorrect, we only allow doing this for down casting.
print("TEST: Test ScalarType Invalid Construction")
try:
    scalarType = compiler.ScalarType(compiler.ScalarTypeCode.f32)
except:
    print("must use '.get' constructor")
# CHECK-LABEL: TEST: Test ScalarType Invalid Construction
# CHECK: must use '.get' constructor

# This is the correct way to create the type.
print("TEST: Test ScalarType Construction")
# CHECK-LABEL: TEST: Test ScalarType Construction
scalarType = compiler.ScalarType.get(compiler.ScalarTypeCode.f32)
print(type(scalarType))
# CHECK: <class 'mlir_tensorrt.compiler._mlir_libs._api.ScalarType'>
print(scalarType)
# CHECK: ScalarType(f32)
print("isinstance = ", compiler.ScalarType.isinstance(scalarType))
# CHECK: isinstance =  True

print("TEST: Test ScalarType Copy")
# CHECK-LABEL: TEST: Test ScalarType Copy
scalarType = compiler.ScalarType(scalarType)
print(type(scalarType))
# CHECK: <class 'mlir_tensorrt.compiler._mlir_libs._api.ScalarType'>
print(scalarType)
# CHECK: ScalarType(f32)

print("TEST: Test ScalarType Cast Up")
# CHECK-LABEL: TEST: Test ScalarType Cast Up
upType = compiler.Type(scalarType)
print(type(upType))
# CHECK: <class 'mlir_tensorrt.compiler._mlir_libs._api.Type'>
print(upType)
# CHECK: <mlir_tensorrt.compiler._mlir_libs._api.Type object

print("TEST: Test ScalarType Cast Down")
# CHECK-LABEL: TEST: Test ScalarType Cast Down
downType = compiler.ScalarType(upType)
print(type(downType))
print(downType)
# CHECK: <class 'mlir_tensorrt.compiler._mlir_libs._api.ScalarType'>
# CHECK: ScalarType(f32)

# This is incorrect, we only allow doing this for down casting.
print("TEST: Test MemRefType Invalid Construction")
try:
    memrefType = compiler.MemRefType(
        [1, 2], compiler.ScalarTypeCode.i32, compiler.PointerType.host
    )
except:
    print("must use '.get' constructor")
# CHECK-LABEL: TEST: Test MemRefType Invalid Construction
# CHECK: must use '.get' constructor

# This is the correct way to create the type.
print("TEST: Test MemRefType Construction")
# CHECK-LABEL: TEST: Test MemRefType Construction
memrefType = compiler.MemRefType.get(
    [1, 2], compiler.ScalarTypeCode.i32, compiler.PointerType.host
)
print(type(memrefType))
# CHECK: <class 'mlir_tensorrt.compiler._mlir_libs._api.MemRefType'>
print(memrefType)
# CHECK: MemRefType()
print("isinstance = ", compiler.MemRefType.isinstance(memrefType))
# CHECK: isinstance =  True

print("TEST: Test MemRefType Copy")
# CHECK-LABEL: TEST: Test MemRefType Copy
memrefType = compiler.MemRefType(memrefType)
print(type(memrefType))
# CHECK: <class 'mlir_tensorrt.compiler._mlir_libs._api.MemRefType'>
print(memrefType)
# CHECK: MemRefType()

print("TEST: Test MemRefType Cast Up")
# CHECK-LABEL: TEST: Test MemRefType Cast Up
upType = compiler.Type(memrefType)
print(type(upType))
# CHECK: <class 'mlir_tensorrt.compiler._mlir_libs._api.Type'>
print(upType)
# CHECK: <mlir_tensorrt.compiler._mlir_libs._api.Type object

print("TEST: Test MemRefType Cast Down")
# CHECK-LABEL: TEST: Test MemRefType Cast Down
downType = compiler.MemRefType(upType)
print(type(downType))
# CHECK: <class 'mlir_tensorrt.compiler._mlir_libs._api.MemRefType'>
print(downType)
# CHECK: MemRefType()
