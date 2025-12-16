configure_file(
  setup.py
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages/mlir_tensorrt_jax/setup.py
  @ONLY
  )

set(MLIR_TRT_JAX_PLUGIN_SOURCES
  README.md
  pyproject.toml
  mlir_tensorrt_jax/__init__.py
  mlir_tensorrt_jax/mtrt_ops/__init__.py
  mlir_tensorrt_jax/mtrt_ops/quantize.py
  mlir_tensorrt_jax/mtrt_ops/dequantize.py
  mlir_tensorrt_jax/mtrt_ops/dynamic_quantize.py
  mlir_tensorrt_jax/mtrt_ops/utils.py
  mlir_tensorrt_jax/flashinfer/__init__.py
  mlir_tensorrt_jax/flashinfer/decode.py
  mlir_tensorrt_jax/flashinfer/gemm.py
  mlir_tensorrt_jax/flashinfer/prefill.py
  mlir_tensorrt_jax/flashinfer/utils.py
  )

declare_mlir_python_sources(MLIRTensorRTJAXPluginSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    ${MLIR_TRT_JAX_PLUGIN_SOURCES}
)

add_mlir_python_modules(MLIRTensorRTJAXPluginModules
  ROOT_PREFIX "${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages/mlir_tensorrt_jax"
  INSTALL_PREFIX "python_packages/mlir_tensorrt_jax"
  DECLARED_SOURCES
    MLIRTensorRTJAXPluginSources
  )
