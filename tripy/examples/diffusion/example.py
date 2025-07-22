#
# SPDX-FileCopyrightText: Copyright (c) 2025-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import nvtripy as tp

from transformers import CLIPTokenizer
from examples.diffusion.models.clip_model import CLIPConfig
from examples.diffusion.models.model import StableDiffusion, StableDiffusionConfig
from examples.diffusion.weight_loader import load_from_diffusers


def compile_model(model, inputs, engine_path, verbose=False):
    if os.path.exists(engine_path):
        if verbose:
            print(f"[I] Loading cached engines from {engine_path}...")
        return tp.Executable.load(engine_path)

    if verbose:
        name = model.__class__.__name__ if isinstance(model, tp.Module) else model.__name__
        print(f"[I] Compiling {name}...", end=" ", flush=True)
        compile_start_time = time.perf_counter()

    compiled_model = tp.compile(model, args=inputs, optimization_level=5)
    compiled_model.save(engine_path)

    if verbose:
        compile_end_time = time.perf_counter()
        print(f"saved engine to {engine_path}.")
        print(f"took {compile_end_time - compile_start_time} seconds.")

    return compiled_model


def compile_clip(model, engine_path, dtype=tp.int32, verbose=False):
    inputs = (tp.InputInfo((1, 77), dtype=dtype),)
    return compile_model(model, inputs, engine_path, verbose=verbose)


def compile_unet(model, engine_path, dtype, verbose=False):
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
    return compile_model(model, inputs, engine_path, verbose=verbose)


def compile_vae(model, engine_path, dtype, verbose=False):
    inputs = (tp.InputInfo((1, 4, 64, 64), dtype=dtype),)
    return compile_model(model, inputs, engine_path, verbose=verbose)


# equivalent to LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000, dtype=torch.float32):
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, n_training_steps, dtype=dtype, device="cuda") ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod


def run_diffusion_loop(model, unconditional_context, context, latent, steps, guidance, dtype, verbose=False):
    torch_dtype = torch.float16 if dtype == tp.float16 else torch.float32
    idx_timesteps = list(range(1, 1000, 1000 // steps))
    num_timesteps = len(idx_timesteps)
    timesteps = torch.tensor(idx_timesteps, dtype=torch_dtype, device="cuda")
    guidance = torch.tensor([guidance], dtype=torch_dtype, device="cuda")

    if verbose:
        print(f"[I] Running diffusion for {steps} timesteps...")
    alphas = get_alphas_cumprod(dtype=torch_dtype)[idx_timesteps]
    alphas_prev = torch.cat((torch.tensor([1.0], dtype=torch_dtype, device="cuda"), alphas[:-1]))

    if verbose:
        iterator = tqdm(list(range(num_timesteps))[::-1])
    else:
        iterator = list(range(num_timesteps))[::-1]

    for index in iterator:
        latent = model(
            unconditional_context,
            context,
            latent,
            tp.Tensor(timesteps[index : index + 1]),
            tp.Tensor(alphas[index : index + 1]),
            tp.Tensor(alphas_prev[index : index + 1]),
            tp.Tensor(guidance),
        )

    return latent


def save_image(image, args):
    if args.out:
        filename = args.out
    else:
        filename = (
            f"{'tp'}-"
            f"{'fp32' if args.fp32 else 'fp16'}-"
            f"{args.prompt[:10].replace(' ', '_')}-"
            f"steps{args.steps}-"
            f"seed{args.seed if args.seed else 'rand'}-"
            f"{int(time.time())}.png"
        )
        filename = os.path.join("output", filename)

    # Save image
    print(f"[I] Saving image to {filename}")
    if not os.path.isdir(os.path.dirname(filename)):
        print(f"[I] Creating '{os.path.dirname(filename)}' directory.")
        os.makedirs(os.path.dirname(filename))
    image.save(filename)


def tripy_diffusion(args):
    run_start_time = time.perf_counter() if args.verbose else None

    dtype, torch_dtype = (tp.float32, torch.float32) if args.fp32 else (tp.float16, torch.float16)

    # Check which engines we need to compile (if any)
    clip_path = os.path.join(args.engine_dir, "clip_executable.tpymodel")
    unet_path = os.path.join(args.engine_dir, "unet_executable.tpymodel")
    vae_path = os.path.join(args.engine_dir, "vae_executable.tpymodel")

    clip_exists = os.path.exists(clip_path)
    unet_exists = os.path.exists(unet_path)
    vae_exists = os.path.exists(vae_path)

    model = StableDiffusion(StableDiffusionConfig(dtype=dtype))
    if not (clip_exists and unet_exists and vae_exists):
        if args.verbose:
            print("[I] Loading model weights...", flush=True)
        load_from_diffusers(model, dtype, args.hf_token)

    if not os.path.isdir(args.engine_dir):
        os.mkdir(args.engine_dir)

    # Load existing engines if they exist, otherwise compile and save them
    clip_compiled = compile_clip(model.text_encoder, engine_path=clip_path, verbose=args.verbose)
    unet_compiled = compile_unet(model, engine_path=unet_path, dtype=dtype, verbose=args.verbose)
    vae_compiled = compile_vae(model.decode, engine_path=vae_path, dtype=dtype, verbose=args.verbose)

    # Run through CLIP to get context from prompt
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    torch_prompt = tokenizer(
        args.prompt, padding="max_length", max_length=CLIPConfig.max_seq_len, truncation=True, return_tensors="pt"
    )
    prompt = tp.Tensor(torch_prompt.input_ids.to(torch.int32).to("cuda"))
    if args.verbose:
        print(f"[I] Got tokenized prompt.")
    torch_unconditional_prompt = tokenizer(
        [""], padding="max_length", max_length=CLIPConfig.max_seq_len, return_tensors="pt"
    )
    unconditional_prompt = tp.Tensor(torch_unconditional_prompt.input_ids.to(torch.int32).to("cuda"))
    if args.verbose:
        print(f"[I] Got unconditional tokenized prompt.")

    if args.verbose:
        print("[I] Getting CLIP conditional and unconditional context...", end=" ")
    clip_run_start = time.perf_counter() if args.verbose else None
    context = clip_compiled(prompt)
    unconditional_context = clip_compiled(unconditional_prompt)
    if args.verbose:
        tp.default_stream().synchronize()
        clip_run_end = time.perf_counter()
        print(f"took {clip_run_end - clip_run_start} seconds.")
    else:
        clip_run_start = None
        clip_run_end = None

    # Backbone of diffusion - the UNet
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch_latent = torch.randn((1, 4, 64, 64), dtype=torch_dtype, device="cuda")
    latent = tp.Tensor(torch_latent)

    diffusion_run_start = time.perf_counter() if args.verbose else None
    latent = run_diffusion_loop(
        unet_compiled, unconditional_context, context, latent, args.steps, args.guidance, dtype, verbose=args.verbose
    )
    if args.verbose:
        tp.default_stream().synchronize()
        diffusion_run_end = time.perf_counter()
        print(f"[I] Finished diffusion denoising. Inference took {diffusion_run_end - diffusion_run_start} seconds.")
    else:
        diffusion_run_start = None
        diffusion_run_end = None

    # Upsample latent space to image with autoencoder
    if args.verbose:
        print(f"[I] Decoding latent...", end=" ")
    vae_run_start = time.perf_counter() if args.verbose else None
    x = vae_compiled(latent)
    if args.verbose:
        tp.default_stream().synchronize()
        vae_run_end = time.perf_counter()
        print(f"took {vae_run_end - vae_run_start} seconds.")
    else:
        vae_run_start = None
        vae_run_end = None

    # Evaluate output
    run_end_time = time.perf_counter() if args.verbose else None
    if args.verbose:
        print(f"[I] Full script took {run_end_time - run_start_time} seconds.")

    image_array = np.from_dlpack(tp.copy(x, tp.device("cpu"))).astype(np.uint8, copy=False)
    image = Image.fromarray(image_array)

    return image, [clip_run_start, clip_run_end, diffusion_run_start, diffusion_run_end, vae_run_start, vae_run_end]


def print_summary(denoising_steps, times, verbose=False):
    if not verbose or times is None or None in times:
        return

    stages_ms = [1000 * (times[i + 1] - times[i]) for i in range(0, 6, 2)]
    total_ms = sum(stages_ms)
    print("|-----------------|--------------|")
    print("| {:^15} | {:^12} |".format("Module", "Latency"))
    print("|-----------------|--------------|")
    print("| {:^15} | {:>9.2f} ms |".format("CLIP", stages_ms[0]))
    print("| {:^15} | {:>9.2f} ms |".format("UNet" + " x " + str(denoising_steps), stages_ms[1]))
    print("| {:^15} | {:>9.2f} ms |".format("VAE-Dec", stages_ms[2]))
    print("|-----------------|--------------|")
    print("| {:^15} | {:>9.2f} ms |".format("Pipeline", total_ms))
    print("|-----------------|--------------|")
    print("Throughput: {:.2f} image/s".format(1000.0 / total_ms))


def main():
    default_prompt = "a beautiful photograph of Mt. Fuji during cherry blossom"
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps in diffusion")
    parser.add_argument("--prompt", type=str, default=default_prompt, help="Phrase to render")
    parser.add_argument("--out", type=str, default=None, help="Output filepath")
    parser.add_argument("--fp32", action="store_true", help="Run the model in fp32 precision")
    parser.add_argument("--seed", type=int, help="Set the random latent seed")
    parser.add_argument("--guidance", type=float, default=7.5, help="Prompt strength")
    parser.add_argument(
        "--hf-token", type=str, default="", help="HuggingFace API access token for downloading model checkpoints"
    )
    parser.add_argument("--engine-dir", type=str, default="engines", help="Output directory for Tripy executables")
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose output with timing and progress bars"
    )
    args = parser.parse_args()

    image, times = tripy_diffusion(args)

    save_image(image, args)
    print_summary(args.steps, times, verbose=args.verbose)


if __name__ == "__main__":
    main()
