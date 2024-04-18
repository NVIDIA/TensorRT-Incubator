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
    """
    from quantization import ammo_quantize

    def convert_to_scale(amax, maxbound):
        return amax.float() / maxbound

    def get_submodule(module, attr_name):
        attrs = attr_name.split(".")
        for attr in attrs:
            if isinstance(module, torch.nn.ModuleList):
                module = module[int(attr)]
            elif isinstance(module, torch.nn.ModuleDict):
                module = module[attr]
            else:
                module = getattr(module, attr)
        return module

    print(f"Loading weights from pretrained model: '{model_type}'")

    tripy_state_dict = model.state_dict()

    # Load huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    model_hf = ammo_quantize(model_hf, quant_mode)
    hf_state_dict = model_hf.state_dict()
    # We ignore some of the keys in the HF checkpoint:
    ignored_keys = [".attn.masked_bias", ".attn.bias", "_pre_quant_scale"]
    hf_keys = [key for key in hf_state_dict.keys() if not any(key.endswith(w) for w in ignored_keys)]

    # See https://paperswithcode.com/method/weight-tying for details on why we do this:
    hf_state_dict["transformer.wte.weight"] = hf_state_dict["lm_head.weight"]

    # ammo has transposed the attn weights
    for key in hf_keys:
        weight = hf_state_dict[key]
        if key.endswith("quantizer._amax"):
            # reshape amax tensor for int4 block quantization
            if quant_mode == "int4-weight-only":
                linear = get_submodule(model_hf, key[: -len(".weight_quantizer._amax")])
                weight = weight.reshape((-1, linear.in_features))
            # compute scale
            quantizer = get_submodule(model_hf, key[: -len("._amax")])
            weight = convert_to_scale(weight, quantizer.maxbound).squeeze()
            # convert to tripy's key for scales
            key, _ = key.split("quantizer._amax")
            key += "scale"

        param = tp.Parameter(weight.contiguous())
        if "ln" not in key:
            param = tp.cast(param, dtype)
        tripy_state_dict[key] = param

    model.load_from_state_dict(tripy_state_dict)
    print("Loaded weights to tripy model.")
