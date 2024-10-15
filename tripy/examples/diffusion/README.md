# Implementing Stable Diffusion With Tripy

## Introduction

This example demonstrates how to implement a Stable Diffusion model using Tripy APIs.

It's broken up into three components:

1. `model.py` defines the model using `tripy.Module` and associated APIs. `clip_model.py`, `unet_model.py`, `vae_model.py` implement specific components of the diffusion model. 
2. `weight_loader.py` loads weights from a HuggingFace checkpoint.
3. `example.py` runs the end-to-end example, taking input text as a command-line argument, running inference, and then displaying the generated output.

The model defaults to running in `float32`, but is recommended to run in `float16` by providing the `--fp16` flag if you have less than 20-24 GB of GPU memory (note that normalization layers will still run in `float32` to preserve accuracy).

## Running The Example

1. Install prerequisites:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2. Run the example:

    ```bash
    python3 example.py --seed 0 --steps 50 --prompt "a beautiful photograph of Mt. Fuji during cherry blossom" --fp16 --engine-dir fp16_engines
    ```