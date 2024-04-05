import torch
from transformers import GPT2LMHeadModel

import tripy as tp


def load_weights_from_hf(model, model_type, dtype):
    print(f"Loading weights from pretrained model: '{model_type}'")

    tripy_state_dict = model.state_dict()
    # attention biases are initialized in the model based on block size.
    tripy_keys = [key for key in tripy_state_dict.keys() if not key.endswith(".attn.bias")]

    # Load huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    hf_state_dict = model_hf.state_dict()
    # We ignore some of the keys in the HF checkpoint:
    hf_keys = [
        key for key in hf_state_dict.keys() if not key.endswith(".attn.masked_bias") and not key.endswith(".attn.bias")
    ]
    assert len(hf_keys) == len(tripy_keys), f"Mismatched keys: {hf_keys} != {tripy_keys}"

    # See https://paperswithcode.com/method/weight-tying for details on why we do this:
    hf_state_dict["transformer.wte.weight"] = hf_state_dict["lm_head.weight"]

    transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
    for key in hf_keys:
        weight = hf_state_dict[key]
        if any(key.endswith(w) for w in transposed):
            with torch.no_grad():
                weight = hf_state_dict[key].t().contiguous()
        param = tp.Parameter(weight)
        if "ln" not in key:
            param = tp.cast(param, dtype)
        tripy_state_dict[key] = param

    model.load_from_state_dict(tripy_state_dict)


def load_quant_weights_from_hf(model, model_type, dtype, quant_mode):
    """
    Loads quantization weights and computes weight scales.
    Only works for int8 weight-only quantization mode for now.
    """
    from quantization import ammo_quantize

    # Value of int8 torch.TensorQuantizer.maxbound
    INT8_MAXBOUND = 127

    print(f"Loading weights from pretrained model: '{model_type}'")

    tripy_state_dict = model.state_dict()

    # Load huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    model_hf = ammo_quantize(model_hf, quant_mode)
    hf_state_dict = model_hf.state_dict()
    # We ignore some of the keys in the HF checkpoint:
    ignored_keys = [".attn.masked_bias", ".attn.bias"]
    hf_keys = [key for key in hf_state_dict.keys() if not any(key.endswith(w) for w in ignored_keys)]

    # See https://paperswithcode.com/method/weight-tying for details on why we do this:
    hf_state_dict["transformer.wte.weight"] = hf_state_dict["lm_head.weight"]

    # ammo has transposed the attn weights
    for key in hf_keys:
        weight = hf_state_dict[key]
        if key.endswith("weight_quantizer._amax"):
            # compute scale
            weight = hf_state_dict[key].float() / INT8_MAXBOUND
            weight = weight.squeeze()
            # convert to tripy's key for scales
            key, _ = key.split("weight_quantizer._amax")
            key += "weight_scale"

        param = tp.Parameter(weight.contiguous())
        if "ln" not in key:
            param = tp.cast(param, dtype)
        tripy_state_dict[key] = param

    model.load_from_state_dict(tripy_state_dict)
    print("Loaded weights to tripy model.")
