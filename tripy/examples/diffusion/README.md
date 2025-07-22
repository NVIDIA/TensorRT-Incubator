# Implementing Stable Diffusion

## Introduction

This example implements a Stable Diffusion model using Tripy.
There are 3 components:

1. `model.py` defines the model using `tripy.Module` and associated APIs. `clip_model.py`, `unet_model.py`, `vae_model.py` implement specific components of the diffusion model. All files live under the `models/` folder.
2. `weight_loader.py` loads weights from a HuggingFace checkpoint.
3. `example.py` runs the end-to-end example, taking input text as a command-line argument, running inference, and then saves the generated output.

The model defaults to running in `float16`, but you can increase the precision by using the `--fp32` flag.

## Running The Example

1. Install prerequisites:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2. Run the example:

    ```bash
    python3 example.py --seed 420 --steps 50 --prompt "a beautiful photograph of Mt. Fuji during cherry blossom" --engine-dir fp16_engines --verbose
    ```

3. **[Optional]** Compare with torch reference to verify accuracy:
    ```bash
    python3 compare_images.py
    ```

    <!--
    Tripy: TEST: EXPECTED_STDOUT Start
    ```
    .*Passed: Images are similar.*SSIM.*0\.8
    ```
    Tripy: TEST: EXPECTED_STDOUT End
    -->
