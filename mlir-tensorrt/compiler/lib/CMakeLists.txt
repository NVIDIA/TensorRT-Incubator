add_subdirectory(Analysis)

add_subdirectory(Interfaces)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Transforms)
add_subdirectory(Backends)

# Compiler must come before CAPI
add_subdirectory(Compiler)

# CAPI must come last
add_subdirectory(CAPI)
