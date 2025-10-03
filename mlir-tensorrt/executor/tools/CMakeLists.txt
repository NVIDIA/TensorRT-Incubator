function(add_mlir_executor_tool target)
  mtrt_add_tool(${target}
    IGNORE_LINK_MTRT
    ${ARGN}
    )
endfunction()

add_subdirectory(executor-opt)
add_subdirectory(executor-translate)
add_subdirectory(executor-runner)
