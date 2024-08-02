include(CheckCXXSourceCompiles)

#-------------------------------------------------------------------------------------
# Set project required compilation definitions for the given target.
#-------------------------------------------------------------------------------------
function(_mtrt_set_target_compile_defs target)
  set(compile_defs )
  if(MLIR_TRT_ENABLE_HLO)
    list(APPEND compile_defs MLIR_TRT_ENABLE_HLO)
  endif()
  if(MLIR_TRT_TARGET_TENSORRT)
    list(APPEND compile_defs MLIR_TRT_TARGET_TENSORRT)
  endif()
  if(MLIR_TRT_ENABLE_NCCL)
    list(APPEND compile_defs MLIR_TRT_ENABLE_NCCL)
  endif()
  if(MLIR_TRT_TARGET_CPP)
    list(APPEND compile_defs MLIR_TRT_TARGET_CPP)
  endif()
  if(MLIR_TRT_TARGET_LUA)
    list(APPEND compile_defs MLIR_TRT_TARGET_LUA)
  endif()
  if(MLIR_TRT_ENABLE_EXECUTOR)
    list(APPEND compile_defs MLIR_TRT_ENABLE_EXECUTOR)
  endif()
  if(MLIR_TRT_ENABLE_PYTHON)
    list(APPEND compile_defs MLIR_TRT_ENABLE_PYTHON)
  endif()
  if(MLIR_TRT_ENABLE_NVTX)
    list(APPEND compile_defs MLIR_TRT_ENABLE_NVTX)
  endif()
  target_compile_definitions(${target} PRIVATE
    ${compile_defs}
    $<$<BOOL:${MLIR_TRT_ENABLE_TESTING}>:MLIR_TRT_ENABLE_TESTING>
    )
endfunction()
