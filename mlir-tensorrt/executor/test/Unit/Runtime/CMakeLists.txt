add_mlir_executor_unittest(Int4Tests Int4Tests.cpp)

if(MLIR_EXECUTOR_ENABLE_GPU_INTEGRATION_TESTS)
    add_subdirectory(LuaRuntime)
endif()
