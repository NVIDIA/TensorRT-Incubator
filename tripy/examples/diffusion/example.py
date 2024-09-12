#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse, os
from tqdm import tqdm
from PIL import Image
import time

import torch
import cupy as cp
import numpy as np
import tripy as tp

from transformers import CLIPTokenizer
from examples.diffusion.clip_model import CLIPConfig
from examples.diffusion.model import StableDiffusion, StableDiffusionConfig, get_alphas_cumprod
from examples.diffusion.weight_loader import load_from_diffusers


def compile_model(model, inputs, verbose=False):
    if verbose:
        name = model.__class__.__name__ if isinstance(model, tp.Module) else model.__name__
        print(f"[I] Compiling {name}...", end=' ', flush=True)
        compile_start_time = time.perf_counter()

    compiler = tp.Compiler(model)
    compiled_model = compiler.compile(*inputs)

    if verbose:
        compile_end_time = time.perf_counter()
        print(f"took {compile_end_time - compile_start_time} seconds.")
    
    return compiled_model


def compile_clip(model, dtype=tp.int32, verbose=False):
    inputs = (tp.InputInfo((1, 77), dtype=dtype),)
    return compile_model(model, inputs, verbose=verbose)


def compile_unet(model, dtype, verbose=False):
    unconditional_context_shape = (1, 77, 768)
    conditional_context_shape = (1, 77, 768)
    latent_shape = (1, 4, 64, 64)
    inputs = (
        tp.InputInfo(unconditional_context_shape, dtype=dtype),
        tp.InputInfo(conditional_context_shape, dtype=dtype),
        tp.InputInfo(latent_shape, dtype=dtype),
        tp.InputInfo((1,), dtype=dtype),
        tp.InputInfo((1,), dtype=dtype),
        tp.InputInfo((1,), dtype=dtype),
        tp.InputInfo((1,), dtype=dtype),
    )
    return compile_model(model, inputs, verbose=verbose)


def compile_vae(model, dtype, verbose=False):
    inputs = (tp.InputInfo((1, 4, 64, 64), dtype=dtype),)
    return compile_model(model, inputs, verbose=verbose)


def run_diffusion_loop(model, unconditional_context, context, latent, steps, guidance, dtype):
    timesteps = list(range(1, 1000, 1000 // steps))
    print(f"[I] Running diffusion for {steps} timesteps...")
    alphas = get_alphas_cumprod(dtype=dtype)[tp.Tensor(timesteps)]
    alphas_prev = tp.concatenate([tp.Tensor([1.0], dtype=dtype), alphas[:-1]], dim=0)

    for index, timestep in (t := tqdm(list(enumerate(timesteps))[::-1])):
        t.set_description("idx: %1d, timestep: %3d" % (index, timestep))
        tid = tp.Tensor([index])
        latent = model(
            unconditional_context,
            context,
            latent,
            tp.Tensor([timestep], dtype=dtype),
            alphas[tid],
            alphas_prev[tid],
            tp.Tensor([guidance], dtype=dtype),
        )
    return latent


def tripy_diffusion(args):
    run_start_time = time.perf_counter()

    dtype, torch_dtype = (tp.float16, torch.float16) if args.fp16 else (tp.float32, torch.float32)

    if os.path.isdir(args.engine_dir):
        print("[I] Loading cached engines from disk...")
        clip_compiled = tp.Executable.load(os.path.join("engines", "clip_executable.json"))
        unet_compiled = tp.Executable.load(os.path.join("engines", "unet_executable.json"))
        vae_compiled = tp.Executable.load(os.path.join("engines", "vae_executable.json"))
    else:
        model = StableDiffusion(StableDiffusionConfig(dtype=dtype))
        print("[I] Loading model weights...", flush=True)
        load_from_diffusers(model, dtype, args.hf_token, debug=True)
        clip_compiled = compile_clip(model.cond_stage_model.transformer.text_model, verbose=True)
        unet_compiled = compile_unet(model, dtype, verbose=True)
        vae_compiled = compile_vae(model.decode, dtype, verbose=True)
        
        os.mkdir(args.engine_dir)
        print(f"[I] Saving engines to {args.engine_dir}...")
        clip_compiled.save(os.path.join("engines", "clip_executable.json"))
        unet_compiled.save(os.path.join("engines", "unet_executable.json"))
        vae_compiled.save(os.path.join("engines", "vae_executable.json"))

    # Run through CLIP to get context from prompt
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    torch_prompt = tokenizer(args.prompt, padding="max_length", max_length=CLIPConfig.max_seq_len, truncation=True, return_tensors="pt")
    prompt = tp.Tensor(torch_prompt.input_ids.to(torch.int32).to("cuda"))
    print(f"[I] Got tokenized prompt.")
    torch_unconditional_prompt = tokenizer([""], padding="max_length", max_length=CLIPConfig.max_seq_len, return_tensors="pt")
    unconditional_prompt = tp.Tensor(torch_unconditional_prompt.input_ids.to(torch.int32).to("cuda"))
    print(f"[I] Got unconditional tokenized prompt.")

    print("[I] Getting CLIP conditional and unconditional context...", end=" ")
    clip_run_start = time.perf_counter()
    context = clip_compiled(prompt)
    unconditional_context = clip_compiled(unconditional_prompt)
    clip_run_end = time.perf_counter()
    print(f"took {clip_run_end - clip_run_start} seconds.")

    # Backbone of diffusion - the UNet
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch_latent = torch.randn((1, 4, 64, 64), dtype=torch_dtype).to("cuda")
    latent = tp.Tensor(torch_latent)

    diffusion_run_start = time.perf_counter()
    latent = run_diffusion_loop(unet_compiled, unconditional_context, context, latent, args.steps, args.guidance, dtype)
    diffusion_run_end = time.perf_counter()
    print(f"[I] Finished diffusion denoising. Inference took {diffusion_run_end - diffusion_run_start} seconds.")

    # Upsample latent space to image with autoencoder
    print(f"[I] Decoding latent...", end=" ")
    vae_run_start = time.perf_counter()
    x = vae_compiled(latent)
    vae_run_end = time.perf_counter()
    print(f"took {vae_run_end - vae_run_start} seconds.")

    # Evaluate output
    x.eval()
    run_end_time = time.perf_counter()
    print(f"[I] Full script took {run_end_time - run_start_time} seconds.")

    # Save image
    image = Image.fromarray(cp.from_dlpack(x).get().astype(np.uint8, copy=False))
    print(f"[I] Saving {args.out}")
    if not os.path.isdir("output"):
        print("[I] Creating 'output' directory.")
        os.mkdir("output")
    image.save(args.out)

    return image, [clip_run_start, clip_run_end, diffusion_run_start, diffusion_run_end, vae_run_start, vae_run_end]

# referenced from https://huggingface.co/blog/stable_diffusion
def hf_diffusion(args):
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
    from tqdm.auto import tqdm

    run_start_time = time.perf_counter()

    dtype = torch.float16 if args.fp16 else torch.float32
    model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if args.fp16 else {} 

    # Initialize models
    model_id = "KiwiXR/stable-diffusion-v1-5" 
    
    print("[I] Loading models...")
    hf_tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    hf_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to("cuda")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", use_auth_token=args.hf_token, **model_opts).to("cuda")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_auth_token=args.hf_token, **model_opts).to("cuda")
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    # Run through CLIP to get context from prompt
    print("[I] Starting tokenization and running clip...", end=" ")
    clip_run_start = time.perf_counter()
    text_input = hf_tokenizer(args.prompt, padding="max_length", max_length=hf_tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda")
    max_length = text_input.input_ids.shape[-1] # 77
    uncond_input = hf_tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt").to("cuda")
    text_embeddings = hf_encoder(text_input.input_ids, output_hidden_states=True)[0]
    uncond_embeddings = hf_encoder(uncond_input.input_ids)[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype)
    clip_run_end = time.perf_counter()
    print(f"took {clip_run_end - clip_run_start} seconds.")

    # Backbone of diffusion - the UNet
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch_latent = torch.randn((1, 4, 64, 64), dtype=dtype).to("cuda")
    torch_latent *= scheduler.init_noise_sigma
    
    scheduler.set_timesteps(args.steps)

    diffusion_run_start = time.perf_counter()
    print(f"[I] Running diffusion for {args.steps} timesteps...")
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([torch_latent] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        torch_latent = scheduler.step(noise_pred, t, torch_latent).prev_sample

    diffusion_run_end = time.perf_counter()
    print(f"[I] Finished diffusion denoising. Inference took {diffusion_run_end - diffusion_run_start} seconds.")

    # Upsample latent space to image with autoencoder
    print(f"[I] Decoding latent...", end=" ")
    vae_run_start = time.perf_counter()
    torch_latent = 1 / 0.18215 * torch_latent
    with torch.no_grad():
        image = vae.decode(torch_latent).sample
    vae_run_end = time.perf_counter()
    print(f"took {vae_run_end - vae_run_start} seconds.")

    # Evaluate Output
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    image = pil_images[0]

    run_end_time = time.perf_counter()
    print(f"[I] Full script took {run_end_time - run_start_time} seconds.")

    # Save image
    print(f"[I] Saving {args.out}")
    if not os.path.isdir("output"):
        print("[I] Creating 'output' directory.")
        os.mkdir("output")
    image.save(args.out)
    return image, [clip_run_start, clip_run_end, diffusion_run_start, diffusion_run_end, vae_run_start, vae_run_end]

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


# TODO: Add torch compilation modes
# TODO: Add Timing context
def main():
    default_prompt = "a horse sized cat eating a bagel"
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of denoising steps in diffusion")
    parser.add_argument("--prompt", type=str, default=default_prompt, help="Phrase to render")
    parser.add_argument("--out", type=str, default=os.path.join("output", "rendered.png"), help="Output filename")
    parser.add_argument("--fp16", action="store_true", help="Cast the weights to float16")
    parser.add_argument("--timing", action="store_true", help="Print timing per step")
    parser.add_argument("--seed", type=int, help="Set the random latent seed")
    parser.add_argument("--guidance", type=float, default=7.5, help="Prompt strength")
    parser.add_argument('--torch-inference', action='store_true', help="Run inference with PyTorch (eager mode) instead of TensorRT.")
    parser.add_argument('--hf-token', type=str, default='', help="HuggingFace API access token for downloading model checkpoints")
    parser.add_argument('--engine-dir', type=str, default='engines', help="Output directory for TensorRT engines")
    args = parser.parse_args()

    if args.torch_inference:
        _, times = hf_diffusion(args)
        print_summary(args.steps, times)
    else:
        _, times = tripy_diffusion(args)
        print_summary(args.steps, times)

if __name__ == "__main__":
    main()