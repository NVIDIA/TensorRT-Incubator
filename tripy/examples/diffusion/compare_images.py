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
from skimage.metrics import structural_similarity
import glob


def load_reference_image(image_path, verbose=False):
    """Load reference image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    if verbose:
        print(f"[I] Loading reference image from {image_path}")
    return Image.open(image_path)


def load_tripy_image(image_path, verbose=False):
    """Load tripy image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Tripy image not found: {image_path}")

    if verbose:
        print(f"[I] Loading tripy image from {image_path}")
    return Image.open(image_path)


def find_latest_image_in_output(output_dir="output", verbose=False):
    """Find the most recent image in the output directory."""
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Look for PNG files in the output directory
    pattern = os.path.join(output_dir, "*.png")
    image_files = glob.glob(pattern)

    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {output_dir}")

    image_files.sort(key=os.path.getmtime, reverse=True)

    if verbose:
        print(f"[I] Found {len(image_files)} images in {output_dir}")
        print(f"[I] Using most recent image: {image_files[0]}")

    return image_files[0]


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

    # Image loading options
    parser.add_argument(
        "--tripy-image",
        type=str,
        default=None,
        help="Path to tripy output image to compare. If not specified, will use the most recent image in output/ directory",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="assets/torch_ref_fp16_fuji_steps50_seed420.png",
        help="Path to reference image file to compare against",
    )

    parser.add_argument("--threshold", type=float, default=0.80, help="SSIM threshold for considering images similar")
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

    # Load tripy image
    try:
        if args.tripy_image:
            tripy_img = load_tripy_image(args.tripy_image, args.verbose)
        else:
            image_path = find_latest_image_in_output(verbose=args.verbose)
            tripy_img = load_tripy_image(image_path, args.verbose)
    except FileNotFoundError as e:
        print(f"[E] {e}")
        return 1

    is_similar = compare_images(tripy_img, reference_img, args.threshold)

    return not is_similar


if __name__ == "__main__":
    exit(main())
