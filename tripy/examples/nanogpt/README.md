# Implementing Nano-GPT With Tripy

## Introduction

This example demonstrates how to implement a [NanoGPT model](https://github.com/karpathy/nanoGPT) using Tripy APIs.

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
    python3 example.py --input-text "What is the answer to life, the universe, and everything?"
    ```

3. **[Optional]** You can also use a fixed seed to ensure predictable outputs each time:

    ```bash
    python3 example.py --input-text "What is the answer to life, the universe, and everything?" --seed=0
    ```

    <!--
    Tripy: TEST: EXPECTED_STDOUT Start
    ```
    (?s).*?What is the answer to life, the universe, and everything\? (How can we know what's real\? How can|The answer to the questions that are asked of us)
    ```
    Tripy: TEST: EXPECTED_STDOUT End
    -->

### Running with Quantization

This section shows how to run this example with different quantization modes.

In `quantization.py`, we use `nvidia-modelopt` to quantize the pytorch GPT model, and then calibrate the quantization parameters.
Then the quantization parameters are converted to scales and loaded into the Tripy model by
`load_quant_weights_from_hf` in [`weight_loader.py`](./weight_loader.py).

To run with a quantization mode, pass `--quant-mode` to `example.py`. The supported modes are:

1. Weight-only int8 quantization:

    ```bash
    python3 example.py --input-text "What is the answer to life, the universe, and everything?" --seed=0 --quant-mode int8-weight-only
    ```
    <!--
    Tripy: TEST: EXPECTED_STDOUT Start
    ```
    (?s).*?What is the answer to life, the universe, and everything\? (The answer to the questions that|How is life possible, what is the meaning of|How can)
    ```
    Tripy: TEST: EXPECTED_STDOUT End
    -->

2. Weight-only int4 quantization:

    *Note: `int4` quantization may result in poor accuracy for this model.*
        *We include it here primarily to demonstrate the workflow.*

    ```bash
    python3 example.py --input-text "What is the answer to life, the universe, and everything?" --seed=0 --quant-mode int4-weight-only
    ```
    <!--
    Tripy: TEST: EXPECTED_STDOUT Start
    ```
    (?s).*?What is the answer to life, the universe, and everything\? What is what is what is what is what is
    ```
    Tripy: TEST: EXPECTED_STDOUT End
    -->
