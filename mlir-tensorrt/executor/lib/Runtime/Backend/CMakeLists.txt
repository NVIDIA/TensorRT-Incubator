add_subdirectory(Common)
add_subdirectory(C)
if(MLIR_EXECUTOR_TARGET_LUA)
  add_subdirectory(Lua)
endif()
