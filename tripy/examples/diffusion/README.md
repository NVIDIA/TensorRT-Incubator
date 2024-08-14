# Implementing Stable Diffusion With Tripy

## Introduction

This example demonstrates how to implement a Stable Diffusion model using Tripy APIs.

It's broken up into three components:

1. `model.py` defines the model using `tripy.Module` and associated APIs.
2. `weight_loader.py` loads weights from a HuggingFace checkpoint.
3. `example.py` runs the end-to-end example, taking input text as a command-line argument,
        running inference, and then displaying the generated output.

The model is currently implemented in `float32`.

## Running The Example

1. Install prerequisites:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2. Run the example:

    ```bash
    python3 example.py --seed 0 --steps 50 --prompt "a beautiful photograph of Mt. Fuji during cherry blossom"
    ```