mtrt_add_tool(mlir-tensorrt-opt
   mlir-tensorrt-opt.cpp

   DEPENDS
   ${MLIR_TENSORRT_LIBS}

   SUPPORT_PLUGINS

   LINK_LIBS PRIVATE
   ${MLIR_TENSORRT_LIBS}
   MLIR_LIBS PRIVATE
   MLIROptLib
)

# Manually link the test libraries since `mtrt_add_tool` will ignore LINK_LIBS
# when MLIR_TRT_LINK_MTRT_DYLIB is enabled. The test libraries are meant to be
# linked directly since they account for MLIR_TRT_LINK_MTRT_DYLIB as well.
# The exception to this are test libraries defined outside the `compiler` project.
# These are linked statically against MLIR, so instead
# we should link against the object target here.

# Link against all compiler test libraries.
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE TEST_LIBS OUT_VAR test_libs)
mtrt_get_project_targets(TensorRTDialect LIBRARY_TYPE TEST_LIBS OUT_VAR mtrtd_test_libs)
target_link_libraries(mlir-tensorrt-opt PRIVATE ${test_libs} ${mtrtd_test_libs})



export_executable_symbols_for_plugins(mlir-tensorrt-opt)
