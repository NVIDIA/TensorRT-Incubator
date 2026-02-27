from .quantize import mtrt_quantize
from .dequantize import mtrt_dequantize
from .dynamic_quantize import mtrt_dynamic_quantize
from .nvtx_range import mtrt_nvtx_push, mtrt_nvtx_pop, mtrt_nvtx_annotate, NVTX_COLOR

try:
    from .dot_product_attention import mtrt_fused_attention
except ImportError:
    pass


def register_all_lowerings():
    """Register all MLIR lowerings for the mlir_tensorrt platform.

    This should be called after the mlir_tensorrt plugin has been initialized.
    """
    from .quantize import register_quantize_lowering
    from .dequantize import register_dequantize_lowering
    from .dynamic_quantize import register_dynamic_quantize_lowering

    try:
        from .dot_product_attention import register_dot_product_attention_lowering

        register_dot_product_attention_lowering()
    except ImportError:
        pass

    from .nvtx_range import register_nvtx_range_lowering

    register_quantize_lowering()
    register_dequantize_lowering()
    register_dynamic_quantize_lowering()
    register_nvtx_range_lowering()
