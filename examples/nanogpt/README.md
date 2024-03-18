# Implementing Nano-GPT With Tripy

## Introduction

This example demonstrates how to implement a NanoGPT model using Tripy APIs.

It's broken up into three components:

1. `model.py` defines the model using `tripy.Module` and associated APIs.
2. `weight_loader.py` loads weights from a HuggingFace checkpoint.
3. `example.py` runs the end-to-end example, taking input text as a command-line argument,
        running inference, and then displaying the generated output.

The model is implemented in `float16`, except `LayerNorm` modules in `float32`
for expected accuracy.

## Running The Example

1. Install prerequisites:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2. Run the example:

    ```bash
    python3 example.py --input-text "Do people really like using ONNX?"
    ```

3. **[Optional]** You also use a fixed seed to ensure predictable outputs each time:

    ```bash
    python3 example.py --input-text "Do people really like using ONNX?" --seed=1
    ```

    <!-- Tripy Test: EXPECTED_STDOUT Start -->
    <!--
    ```
    Loading weights from pretrained model: 'gpt2'
    Do people really like using ONNX?

    This is something that I'm very excited
    ```
     -->
    <!-- Tripy Test: EXPECTED_STDOUT End -->
