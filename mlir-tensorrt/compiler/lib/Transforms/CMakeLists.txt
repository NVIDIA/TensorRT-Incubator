# Targets in this directory should contain only passes/transforms that do not
# depend on dialects defined in this project -- they can only depend on upstream
# dialects.
if(MLIR_TRT_ENABLE_HLO)
  add_subdirectory(StablehloInputPreprocessing)
  add_subdirectory(StablehloRaiseQDQ)
  add_subdirectory(StablehloMatchers)
endif()

add_subdirectory(DropNestedModules)
add_subdirectory(DuplicateFunctionElimination)
add_subdirectory(SCFDetensorizeLoops)
add_subdirectory(MemRefCastElimination)
