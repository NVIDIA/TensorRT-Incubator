import torch
import tripy as tp

from diffusers import StableDiffusionPipeline

def load_weights_from_hf(model, hf_model, dtype, debug=False):
    tripy_state_dict = model.state_dict()
    tripy_keys = tripy_state_dict.keys()

    hf_state_dict = hf_model.state_dict()
    hf_keys = hf_state_dict.keys()
    
    assert_msg = f"Mismatched keys: {hf_keys} != {tripy_keys}"
    if debug and len(hf_keys) != len(tripy_keys):
        print("\nERROR: unused weights in HF_state_dict:\n", sorted(list(hf_keys - tripy_keys)))
        print("\nERROR: unused weights in tripy_state_dict:\n", sorted(list(tripy_keys - hf_keys)))
        assert_msg = f"Mismatched keys between HF and provided model, see above."
    assert len(hf_keys) == len(tripy_keys), assert_msg

    torch_dtype = getattr(torch, dtype.name)
    for key in hf_keys:
        weight = hf_state_dict[key]
        # print(weight.dtype)
        # if "ln" in key or "gn" in key or "norm" in key:
        #     print(f"{key}: {weight.dtype}")
        # if "norm" not in key:
        #     weight = weight.to(torch_dtype)
        # print(f"{key}: {weight.dtype}")
        param = tp.Parameter(weight)
        tripy_state_dict[key.removeprefix("text_model.")] = param

    model.load_from_state_dict(tripy_state_dict)

def load_from_diffusers(model, dtype, debug=False):
    model_id = "CompVis/stable-diffusion-v1-4" #"benjamin-paine/stable-diffusion-v1-5"  #"runwayml/stable-diffusion-v1-5"
    model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if dtype == tp.float16 else {} 
    pipe = StableDiffusionPipeline.from_pretrained(model_id, **model_opts)

    load_weights_from_hf(model.cond_stage_model.transformer.text_model, pipe.text_encoder, dtype, debug=debug)
    load_weights_from_hf(model.model.diffusion_model, pipe.unet, dtype, debug=debug)
    load_weights_from_hf(model.first_stage_model, pipe.vae, dtype, debug=debug)

