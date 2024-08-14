import argparse, tempfile
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import time

import torch
import cupy as cp
import numpy as np

from model import ClipTokenizer, StableDiffusion
from weight_loader import load_from_diffusers
import tripy as tp

def tripy_diffusion(args):
    model = StableDiffusion()
    load_from_diffusers(model, tp.float32, debug=True)

    run_start_time = time.perf_counter()

    # Run through CLIP to get context
    tokenizer = ClipTokenizer()
    prompt = tp.Tensor([tokenizer.encode(args.prompt)])
    print(f"Got tokenized prompt.")
    unconditional_prompt = tp.Tensor([tokenizer.encode("")])
    print(f"Got unconditional tokenized prompt.")

    print("Compiling CLIP model...")
    clip_compile_start_time = time.perf_counter()
    clip_compiler = tp.Compiler(model.cond_stage_model.transformer.text_model)
    clip_text_model = clip_compiler.compile(tp.InputInfo((1, 77), dtype=tp.int32))
    clip_compile_end_time = time.perf_counter()
    print(f"Compilation of CLIP took {clip_compile_end_time - clip_compile_start_time} seconds.")

    print("Getting CLIP context...")
    clip_run_start = time.perf_counter()
    context = clip_text_model(prompt)
    unconditional_context = clip_text_model(unconditional_prompt)
    clip_run_end = time.perf_counter()
    print(f"Got CLIP conditional and unconditional context. Inference took {clip_run_end - clip_run_start} seconds.")

    # Backbone of diffusion - the UNet
    print("Compiling UNet...")
    unet_compile_start_time = time.perf_counter()
    compiler = tp.Compiler(model)
    unconditional_context_shape = (1, 77, 768)
    conditional_context_shape = (1, 77, 768)
    latent_shape = (1, 4, 64, 64)
    compiled_model = compiler.compile(
        tp.InputInfo(unconditional_context_shape, dtype=tp.float32),
        tp.InputInfo(conditional_context_shape, dtype=tp.float32),
        tp.InputInfo(latent_shape, dtype=tp.float32),
        tp.InputInfo((1,), dtype=tp.float32),
        tp.InputInfo((1,), dtype=tp.float32),
        tp.InputInfo((1,), dtype=tp.float32),
        tp.InputInfo((1,), dtype=tp.float32),
    )
    unet_compile_end_time = time.perf_counter()
    print(f"Compilation of UNet took {unet_compile_end_time - unet_compile_start_time} seconds.")

    timesteps = list(range(1, 1000, 1000 // args.steps))
    print(f"Running for {timesteps} timesteps.")
    alphas = model.alphas_cumprod[tp.Tensor(timesteps)]
    alphas_prev = tp.concatenate([tp.Tensor([1.0]), alphas[:-1]], dim=0)

    # start with random noise
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch_latent = torch.randn((1, 4, 64, 64)).to("cuda")
    latent = tp.Tensor(torch_latent)

    def run(model, unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance):
        return model(unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance)

    # This is diffusion
    print("Running diffusion...")
    unet_run_start = time.perf_counter()
    for index, timestep in (t := tqdm(list(enumerate(timesteps))[::-1])):
        t.set_description("idx: %1d, timestep: %3d" % (index, timestep))
        tid = tp.Tensor([index])
        latent = run(
            compiled_model,
            unconditional_context,
            context,
            latent,
            tp.cast(tp.Tensor([timestep]), tp.float32),
            alphas[tid],
            alphas_prev[tid],
            tp.Tensor([args.guidance]),
        )
    unet_run_end = time.perf_counter()
    print(f"Finished running diffusion. Inference took {unet_run_end - unet_run_start} seconds.")

    # Upsample latent space to image with autoencoder
    print("Compiling VAE decoder...")
    vae_compile_start_time = time.perf_counter()
    vae_compiler = tp.Compiler(model.decode)
    vae_decode = vae_compiler.compile(tp.InputInfo((1, 4, 64, 64), dtype=tp.float32))
    vae_compile_end_time = time.perf_counter()
    print(f"Compilation took {vae_compile_end_time - vae_compile_start_time} seconds.")

    print(f"Decoding latent...")
    vae_run_start = time.perf_counter()
    x = vae_decode(latent)
    # x = model.decode(latent)
    vae_run_end = time.perf_counter()
    print(f"Finished decoding latent. Inference took {vae_run_end - vae_run_start} seconds.")

    run_end_time = time.perf_counter()
    x.eval()
    print(f"Full pipeline took {run_end_time - run_start_time} seconds.")

    # save image
    im = Image.fromarray(cp.from_dlpack(x).get().astype(np.uint8, copy=False))
    print(f"saving {args.out}")
    im.save(args.out)
    # Open image.
    if not args.noshow:
        im.show()

def hf_diffusion(args):
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler, AutoencoderKL

    model_id = "runwayml/stable-diffusion-v1-5"  # "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=torch.float32)
    pipe = pipe.to("cuda")
    hf_tokenizer = pipe.tokenizer
    hf_encoder = pipe.text_encoder.to("cuda")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to("cuda")
    
    text_input = hf_tokenizer(args.prompt, padding="max_length", max_length=hf_tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda")
    max_length = text_input.input_ids.shape[-1] # 77
    uncond_input = hf_tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt").to("cuda")

    text_embeddings = hf_encoder(text_input.input_ids, output_hidden_states=True)[0]
    uncond_embeddings = hf_encoder(uncond_input.input_ids)[0]

    from test_acc import check_equal
    del pipe
    model = StableDiffusion()
    load_from_diffusers(model, tp.float32, debug=True)

    run_start_time = time.perf_counter()

    # Run through CLIP to get context
    tokenizer = ClipTokenizer()
    prompt = tp.Tensor([tokenizer.encode(args.prompt)])
    print(f"Got tokenized prompt.")
    unconditional_prompt = tp.Tensor([tokenizer.encode("")])
    print(f"Got unconditional tokenized prompt.")

    print("Compiling CLIP model...")
    clip_compile_start_time = time.perf_counter()
    clip_compiler = tp.Compiler(model.cond_stage_model.transformer.text_model)
    clip_text_model = clip_compiler.compile(tp.InputInfo((1, 77), dtype=tp.int32))
    clip_compile_end_time = time.perf_counter()
    print(f"Compilation of CLIP took {clip_compile_end_time - clip_compile_start_time} seconds.")

    print("Getting CLIP context...")
    clip_run_start = time.perf_counter()
    context = clip_text_model(prompt)
    unconditional_context = clip_text_model(unconditional_prompt)
    clip_run_end = time.perf_counter()
    print(f"Got CLIP conditional and unconditional context. Inference took {clip_run_end - clip_run_start} seconds.")
    check_equal(context, text_embeddings, debug=True)
    check_equal(unconditional_context, uncond_embeddings, debug=True)

    print("DONE")

    # HF DIFFUSERS UNET
    # start with random noise
    # if args.seed is not None:
    torch.manual_seed(0)
    torch_latent = torch.randn((1, 4, 64, 64)).to("cuda")
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    scheduler.set_timesteps(args.steps)

    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([torch_latent] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=999)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, 999, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + args.guidance * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, 999, torch_latent).prev_sample
    print(f"TORCH LATENT: {latents}")

    torch_latent = 1 / 0.18215 * torch_latent
    decoder_out = vae.decode(torch_latent)

def main():
    default_prompt = "a horse sized cat eating a bagel"
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of steps in diffusion")
    parser.add_argument("--prompt", type=str, default=default_prompt, help="Phrase to render")
    parser.add_argument("--out", type=str, default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
    parser.add_argument("--noshow", action="store_true", help="Don't show the image")
    parser.add_argument("--fp16", action="store_true", help="Cast the weights to float16")
    parser.add_argument("--timing", action="store_true", help="Print timing per step")
    parser.add_argument("--seed", type=int, help="Set the random latent seed")
    parser.add_argument("--guidance", type=float, default=7.5, help="Prompt strength")
    parser.add_argument('--torch-inference', action='store_true', help="Run inference with PyTorch (eager mode) instead of TensorRT.")
    args = parser.parse_args()

    if args.torch_inference:
        hf_diffusion(args)
    else:
        tripy_diffusion(args)

if __name__ == "__main__":
    main()