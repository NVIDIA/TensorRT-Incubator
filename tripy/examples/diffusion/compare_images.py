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

import argparse
import os
import numpy as np
from PIL import Image
from argparse import Namespace
from skimage.metrics import structural_similarity

from example import tripy_diffusion


def load_reference_image(image_path, verbose=False):
    """Load reference image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    if verbose:
        print(f"[I] Loading reference image from {image_path}")
    return Image.open(image_path)


def compare_images(tripy_img, reference_img, threshold=0.80):
    """Compare two images using structural similarity index."""
    # Convert both images to grayscale numpy arrays for comparison
    tripy_array = np.array(tripy_img.convert("L"))
    reference_array = np.array(reference_img.convert("L"))

    # Ensure both images have the same dimensions
    if tripy_array.shape != reference_array.shape:
        print(f"[W] Image shape mismatch: tripy {tripy_array.shape} vs reference {reference_array.shape}")
        # Resize reference to match tripy output
        reference_img_resized = reference_img.resize(tripy_img.size, Image.Resampling.LANCZOS)
        reference_array = np.array(reference_img_resized.convert("L"))

    # Calculate structural similarity
    ssim = structural_similarity(tripy_array, reference_array)

    if ssim >= threshold:
        print(f"[I] Passed: Images are similar (SSIM >= {threshold})")
        return True
    else:
        print(f"[I] Failed: Images are not similar enough (SSIM < {threshold})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare tripy diffusion output with a reference image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Reference image argument
    parser.add_argument(
        "--reference",
        type=str,
        default="assets/torch_ref_fp16_fuji_steps50_seed420.png",
        help="Path to reference image file to compare against",
    )

    # Diffusion parameters (matching example.py)
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps in diffusion")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful photograph of Mt. Fuji during cherry blossom",
        help="Phrase to render",
    )
    parser.add_argument("--fp16", action="store_true", help="Cast the weights to float16")
    parser.add_argument("--seed", type=int, help="Set the random latent seed")
    parser.add_argument("--guidance", type=float, default=7.5, help="Prompt strength")
    parser.add_argument(
        "--hf-token", type=str, default="", help="HuggingFace API access token for downloading model checkpoints"
    )
    parser.add_argument("--engine-dir", type=str, default="engines", help="Output directory for TensorRT engines")

    # Comparison parameters
    parser.add_argument("--threshold", type=float, default=0.80, help="SSIM threshold for considering images similar")
    parser.add_argument("--save-output", type=str, default=None, help="Save the tripy output image to this path")
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose output with timing and progress bars"
    )

    args = parser.parse_args()

    # Load reference image
    try:
        reference_img = load_reference_image(args.reference)
    except FileNotFoundError as e:
        print(f"[E] {e}")
        return 1

    # Create args namespace for tripy_diffusion
    tripy_args = Namespace(
        steps=args.steps,
        prompt=args.prompt,
        out=args.save_output,
        fp16=args.fp16,
        seed=args.seed,
        guidance=args.guidance,
        torch_inference=False,
        hf_token=args.hf_token,
        engine_dir=args.engine_dir,
        verbose=args.verbose,
    )

    # Run tripy diffusion
    if args.verbose:
        print(f"[I] Running tripy diffusion with parameters:")
        print(f"    Prompt: {args.prompt}")
        print(f"    Steps: {args.steps}")
        print(f"    FP16: {args.fp16}")
        print(f"    Seed: {args.seed}")
        print(f"    Guidance: {args.guidance}")

    try:
        tripy_img, times = tripy_diffusion(tripy_args)
    except Exception as e:
        print(f"[E] Error running tripy diffusion: {e}")
        return 1

    # Compare images
    is_similar = compare_images(tripy_img, reference_img, args.threshold)

    # Save output if requested
    if args.save_output:
        if not os.path.isdir(os.path.dirname(args.save_output)):
            os.makedirs(os.path.dirname(args.save_output), exist_ok=True)
        tripy_img.save(args.save_output)
        print(f"[I] Saved tripy output to {args.save_output}")

    # Return appropriate exit code
    return 0 if is_similar else 1


if __name__ == "__main__":
    exit(main())
