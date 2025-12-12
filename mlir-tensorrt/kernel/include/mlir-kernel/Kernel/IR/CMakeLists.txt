# LLVM `mlir_tablegen` function depends grabbing include directories
# from the scope, so we must use `include_directories` here.
include_directories(${MLIR_INCLUDE_DIRS})
get_target_property(incDirs MLIRTensorRTCommonIncludes INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${incDirs})

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Enums.td)
mlir_tablegen(Enums.h.inc -gen-enum-decls)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(Attributes.h.inc -gen-attrdef-decls)
mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS Interfaces.td)
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(TypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TypeInterfaces.cpp.inc -gen-type-interface-defs)

mtrt_add_public_tablegen_target(MLIRTensorRTKernelIncGen)
