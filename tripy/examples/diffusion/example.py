import argparse, os
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import time

import torch
import cupy as cp
import numpy as np

from model import ClipTokenizer, StableDiffusion, get_alphas_cumprod
from weight_loader import load_from_diffusers
import tripy as tp


def compile_model(model, inputs, verbose=False):
    if verbose:
        print(f"Compiling {model.__class__.__name__}...", end=' ')
        compile_start_time = time.perf_counter()

    compiler = tp.Compiler(model)
    compiled_model = compiler.compile(*inputs)

    if verbose:
        compile_end_time = time.perf_counter()
        print(f"took {compile_end_time - compile_start_time} seconds.")
    
    return compiled_model


def compile_clip(model, verbose=False):
    inputs = (tp.InputInfo((1, 77), dtype=tp.int32),)
    return compile_model(model, inputs, verbose=verbose)


def compile_unet(model, verbose=False):
    unconditional_context_shape = (1, 77, 768)
    conditional_context_shape = (1, 77, 768)
    latent_shape = (1, 4, 64, 64)
    inputs = (
        tp.InputInfo(unconditional_context_shape, dtype=tp.float32),
        tp.InputInfo(conditional_context_shape, dtype=tp.float32),
        tp.InputInfo(latent_shape, dtype=tp.float32),
        tp.InputInfo((1,), dtype=tp.float32),
        tp.InputInfo((1,), dtype=tp.float32),
        tp.InputInfo((1,), dtype=tp.float32),
        tp.InputInfo((1,), dtype=tp.float32),
    )
    return compile_model(model, inputs, verbose=verbose)


def compile_vae(model, verbose=False):
    inputs = (tp.InputInfo((1, 4, 64, 64), dtype=tp.float32),)
    return compile_model(model, inputs, verbose=verbose)


# def compile_CLIP(model, verbose=False):
#     if verbose:
#         print("Compiling CLIP model...")
#         clip_compile_start_time = time.perf_counter()

#     clip_compiler = tp.Compiler(model)
#     compiled_clip = clip_compiler.compile(tp.InputInfo((1, 77), dtype=tp.int32))

#     if verbose:
#         clip_compile_end_time = time.perf_counter()
#         print(f"Compilation of CLIP took {clip_compile_end_time - clip_compile_start_time} seconds.")

#     return compiled_clip


# def compile_unet(model, verbose=False):
#     if verbose:
#         print("Compiling UNet...")
#         unet_compile_start_time = time.perf_counter()

#     compiler = tp.Compiler(model)
#     unconditional_context_shape = (1, 77, 768)
#     conditional_context_shape = (1, 77, 768)
#     latent_shape = (1, 4, 64, 64)
#     compiled_model = compiler.compile(
#         tp.InputInfo(unconditional_context_shape, dtype=tp.float32),
#         tp.InputInfo(conditional_context_shape, dtype=tp.float32),
#         tp.InputInfo(latent_shape, dtype=tp.float32),
#         tp.InputInfo((1,), dtype=tp.float32),
#         tp.InputInfo((1,), dtype=tp.float32),
#         tp.InputInfo((1,), dtype=tp.float32),
#         tp.InputInfo((1,), dtype=tp.float32),
#     )

#     if verbose:
#         unet_compile_end_time = time.perf_counter()
#         print(f"Compilation of UNet took {unet_compile_end_time - unet_compile_start_time} seconds.")

#     return compiled_model


def run_diffusion_loop(model, unconditional_context, context, latent, steps, guidance):
    timesteps = list(range(1, 1000, 1000 // steps))[::-1]
    # print(f"t: {timesteps}")
    alphas = get_alphas_cumprod()[tp.Tensor(timesteps)]
    alphas_prev = tp.concatenate([tp.Tensor([1.0]), alphas[:-1]], dim=0)
    # print(f"a: {alphas}")
    # print(f"aP: {alphas_prev}")

    # unet_run_start = time.perf_counter()
    for index, timestep in enumerate(timesteps):
        tid = tp.Tensor([index])
        latent = model(
            unconditional_context,
            context,
            latent,
            tp.cast(tp.Tensor([timestep]), tp.float32),
            alphas[tid],
            alphas_prev[tid],
            tp.Tensor([guidance]),
        )
    # unet_run_end = time.perf_counter()
    # print(f"Finished running diffusion. Inference took {unet_run_end - unet_run_start} seconds.")
    return latent


def tripy_diffusion(args):
    model = StableDiffusion()
    load_from_diffusers(model, tp.float32, debug=True)

    run_start_time = time.perf_counter()

    # if os.path.isdir("engines"):
    #     compiled_clip = tp.Executable.load(os.path.join("engines", "clip_executable.json"))
    #     compiled_unet = tp.Executable.load(os.path.join("engines", "unet_executable.json"))
    #     compiled_vae = tp.Executable.load(os.path.join("engines", "vae_executable.json"))
    # else:
    compiled_clip = compile_clip(model.cond_stage_model.transformer.text_model, verbose=True)
    compiled_unet = compile_unet(model, verbose=True)
    compiled_vae = compile_vae(model.decode, verbose=True)
    
    # os.mkdir("engines")
    # compiled_clip.save(os.path.join("engines", "clip_executable.json"))
    # compiled_unet.save(os.path.join("engines", "unet_executable.json"))
    # compiled_vae.save(os.path.join("engines", "vae_executable.json"))

    # Run through CLIP to get context
    tokenizer = ClipTokenizer()
    prompt = tp.Tensor([tokenizer.encode(args.prompt)])
    print(f"Got tokenized prompt.")
    unconditional_prompt = tp.Tensor([tokenizer.encode("")])
    print(f"Got unconditional tokenized prompt.")

    print("Getting CLIP conditional and unconditional context...", end=' ')
    clip_run_start = time.perf_counter()
    context = compiled_clip(prompt)
    unconditional_context = compiled_clip(unconditional_prompt)
    clip_run_end = time.perf_counter()
    print(f"took {clip_run_end - clip_run_start} seconds.")

    # Backbone of diffusion - the UNet
    
    # start with random noise
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch_latent = torch.randn((1, 4, 64, 64)).to("cuda")
    latent = tp.Tensor(torch_latent)

    print(f"Running diffusion loop for {args.steps} steps...", end=' ')

    # compiler = tp.Compiler(run_diffusion_loop)
    # unconditional_context_shape = (1, 77, 768)
    # conditional_context_shape = (1, 77, 768)
    # latent_shape = (1, 4, 64, 64)
    # compiled_diffusion_loop = compiler.compile(
    #     model,
    #     tp.InputInfo(unconditional_context_shape, dtype=tp.float32),
    #     tp.InputInfo(conditional_context_shape, dtype=tp.float32),
    #     tp.InputInfo(latent_shape, dtype=tp.float32),
    #     args.steps, 
    #     args.guidance,
    # )

    timesteps = list(range(1, 1000, 1000 // args.steps))[::-1]
    alphas = get_alphas_cumprod()[tp.Tensor(timesteps)]
    alphas_prev = tp.concatenate([tp.Tensor([1.0]), alphas[:-1]], dim=0)
    tid = tp.Tensor([0])
    diffusion_run_start = time.perf_counter()
    # latent = run_diffusion_loop(compiled_unet, unconditional_context, context, latent, args.steps, args.guidance)
    latent = compiled_unet(
            unconditional_context,
            context,
            latent,
            tp.cast(tp.Tensor([timesteps[0]]), tp.float32),
            alphas[tid],
            alphas_prev[tid],
            tp.Tensor([args.guidance]),
        )
    diffusion_run_end = time.perf_counter()
    print(f"took {diffusion_run_end - diffusion_run_start} seconds.")

    #latent = run_diffusion_loop(compiled_unet, unconditional_context, context, latent, args.steps, args.guidance)

    # timesteps = list(range(1, 1000, 1000 // args.steps))
    # print(f"Running for {timesteps} timesteps.")
    # alphas = model.alphas_cumprod[tp.Tensor(timesteps)]
    # alphas_prev = tp.concatenate([tp.Tensor([1.0]), alphas[:-1]], dim=0)

    # def run(model, unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance):
    #     return model(unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance)

    # # This is diffusion
    # print("Running diffusion...")
    # unet_run_start = time.perf_counter()
    # for index, timestep in (t := tqdm(list(enumerate(timesteps))[::-1])):
    #     t.set_description("idx: %1d, timestep: %3d" % (index, timestep))
    #     tid = tp.Tensor([index])
    #     latent = run(
    #         compiled_unet,
    #         unconditional_context,
    #         context,
    #         latent,
    #         tp.cast(tp.Tensor([timestep]), tp.float32),
    #         alphas[tid],
    #         alphas_prev[tid],
    #         tp.Tensor([args.guidance]),
    #     )
    # unet_run_end = time.perf_counter()
    # print(f"Finished running diffusion. Inference took {unet_run_end - unet_run_start} seconds.")

    # Upsample latent space to image with autoencoder

    print(f"Decoding latent...", end=' ')
    vae_run_start = time.perf_counter()
    x = compiled_vae(latent)
    # x = model.decode(latent)
    vae_run_end = time.perf_counter()
    print(f"took {vae_run_end - vae_run_start} seconds.")

    run_end_time = time.perf_counter()
    x.eval()
    print(f"Full pipeline took {run_end_time - run_start_time} seconds.")

    # save image
    im = Image.fromarray(cp.from_dlpack(x).get().astype(np.uint8, copy=False))
    print(f"saving {args.out}")
    if not os.path.isdir("output"):
        os.mkdir("output")
    im.save(args.out)

    return im, [clip_run_start, clip_run_end, diffusion_run_start, diffusion_run_end, vae_run_start, vae_run_end]

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
    
    run_start_time = time.perf_counter()

    print("Starting tokenization and running clip...", end=" ")
    clip_run_start = time.perf_counter()
    text_input = hf_tokenizer(args.prompt, padding="max_length", max_length=hf_tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda")
    max_length = text_input.input_ids.shape[-1] # 77
    uncond_input = hf_tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt").to("cuda")
    text_embeddings = hf_encoder(text_input.input_ids, output_hidden_states=True)[0]
    uncond_embeddings = hf_encoder(uncond_input.input_ids)[0]
    clip_run_end = time.perf_counter()
    print(f"took {clip_run_end - clip_run_start} seconds.")

    # Diffusion loop with UNet
    if args.seed is not None:
        torch.manual_seed(args.seed)
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

    torch_latent = 1 / 0.18215 * torch_latent
    decoder_out = vae.decode(torch_latent)


def print_summary(denoising_steps, times):
    stages_ms = [1000 * (times[i+1] - times[i]) for i in range(0, 6, 2)]
    total_ms = sum(stages_ms)
    print('|-----------------|--------------|')
    print('| {:^15} | {:^12} |'.format('Module', 'Latency'))
    print('|-----------------|--------------|')
    print('| {:^15} | {:>9.2f} ms |'.format('CLIP', stages_ms[0]))
    print('| {:^15} | {:>9.2f} ms |'.format('UNet'+' x '+str(denoising_steps), stages_ms[1]))
    print('| {:^15} | {:>9.2f} ms |'.format('VAE-Dec', stages_ms[2]))
    print('|-----------------|--------------|')
    print('| {:^15} | {:>9.2f} ms |'.format('Pipeline', total_ms))
    print('|-----------------|--------------|')
    print('Throughput: {:.2f} image/s'.format(1000. / total_ms))


def main():
    default_prompt = "a horse sized cat eating a bagel"
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of denoising steps in diffusion")
    parser.add_argument("--prompt", type=str, default=default_prompt, help="Phrase to render")
    parser.add_argument("--out", type=str, default=os.path.join("output", "rendered.png"), help="Output filename")
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
        _, times = tripy_diffusion(args)
        print_summary(args.steps, times)

if __name__ == "__main__":
    main()