# Implementing Nano-GPT With TriPy

## Introduction

This example demonstrates how to implement a [NanoGPT model](https://github.com/karpathy/nanoGPT) using TriPy APIs.

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

3. **[Optional]** You also use a fixed seed to ensure predictable outputs each time:

    ```bash
    python3 example.py --input-text "What is the answer to life, the universe, and everything?" --seed=1
    ```

    <!-- TriPy: TEST: EXPECTED_STDOUT Start -->
    <!--
    ```
    (?s).*?
    What is the answer to life, the universe, and everything\? The answer to Aquinas, the most important thinker
    ```
     -->
    <!-- TriPy: TEST: EXPECTED_STDOUT End -->

### Running with Quantization

This section shows how to run this example with different quantization modes.

In `quantization.py`, we use `nvidia-modelopt` to quantize the pytorch GPT model, and then calibrate the quantization parameters. Then the quantization parameters are converted to scales and loaded into tripy model in function
`load_quant_weights_from_hf` of `weight_loader.py`.

To run with a quantization mode, pass `--quant-mode` to `example.py`. The supported modes are:

1. weight only int8 quantization:

    ```bash
    python3 example.py --input-text "What is the answer to life, the universe, and everything?" --seed=1 --quant-mode int8-weight-only
    ```
    <!-- TriPy: TEST: EXPECTED_STDOUT Start -->
    <!--
    ```
    (?s).*?
    What is the answer to life, the universe, and everything\? The answer to all of this is: I believe    ```
     -->
    <!-- TriPy: TEST: EXPECTED_STDOUT End -->

2. weight only int4 quantization:

    ```bash
    python3 example.py --input-text "What is the answer to life, the universe, and everything?" --seed=1 --quant-mode int4-weight-only
    ```

<!-- TriPy: TEST: XFAIL Start -->
3. fp8 quantization:

    ```bash
    python3 example.py --input-text "What is the answer to life, the universe, and everything?" --seed=1 --quant-mode fp8
    ```
<!-- TriPy: TEST: XFAIL End -->
