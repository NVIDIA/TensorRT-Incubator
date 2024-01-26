import torch
import re
from transformers import GPT2LMHeadModel
import tripy as tp


def load_weights_from_hf(model, model_type):
    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    print(f"Loading weights from pretrained gpt: {model_type}")

    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [
        k for k in sd_keys if not k.endswith(".attn.bias")
    ]  # Bias are initialized in model init using block size.

    # Load huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]  # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]  # same, just the mask (buffer)
    transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

    sd_hf["transformer.wte.weight"] = sd_hf["lm_head.weight"]  # https://paperswithcode.com/method/weight-tying

    # transformer.h.i weights are stored as transformer.h_i
    converted_list = [re.sub(r"h\.\d+", lambda x: x.group().replace(".", "_"), s) for s in sd_keys_hf]

    for idx, k in enumerate(sd_keys_hf):
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            with torch.no_grad():
                sd[converted_list[idx]] = tp.nn.Parameter(tp.Tensor(sd_hf[k].t().contiguous()))
        else:
            with torch.no_grad():
                sd[converted_list[idx]] = tp.nn.Parameter(tp.Tensor(sd_hf[k]))

    model.load_from_state_dict(sd)
