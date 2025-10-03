mtrt_add_tool(mlir-tensorrt-runner mlir-tensorrt-runner.cpp)

if(MLIR_TRT_LINK_MTRT_DYLIB)
  set(link_deps MTRT)
else()
  set(link_deps MLIRExecutorRunnerLib)
endif()

target_link_libraries(mlir-tensorrt-runner PRIVATE
  ${link_deps}
  )
