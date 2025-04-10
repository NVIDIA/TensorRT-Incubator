# Implementing NanoGPT

## Introduction

This example implements a [NanoGPT model](https://github.com/karpathy/nanoGPT) using Tripy:

1. [`model.py`](./model.py) defines the model as an `nvtripy.Module`.
2. [`weight_loader.py`](./weight_loader.py) loads weights from a HuggingFace checkpoint.
3. [`example.py`](./example.py) runs inference in `float16` on input text and displays the output.


## Running The Example

1. Install prerequisites:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2. Run the example:

    ```bash
    python3 example.py --input-text "What is the answer to life, the universe, and everything?"
    ```

3. **[Optional]** Use a fixed seed for predictable outputs:

    ```bash
    python3 example.py --input-text "What is the answer to life, the universe, and everything?" --seed=0
    ```

    <!--
    Tripy: TEST: EXPECTED_STDOUT Start
    ```
    (?s).*?What is the answer to life, the universe, and everything\? (How can we know what's real\? How can|The answer to the questions that are asked of us|The answer to the questions that are the most difficult)
    ```
    Tripy: TEST: EXPECTED_STDOUT End
    -->

### Running with Quantization

[`quantization.py`](./quantization.py), uses
[NVIDIA TensorRT Model Optimizer](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/1_overview.html)
to quantize the pytorch model.

`load_quant_weights_from_hf` in [`weight_loader.py`](./weight_loader.py) converts the quantization
parameters to scales and loads them into the Tripy model.

Use `--quant-mode` in `example.py` to enable quantization. Supported modes:

- Weight-only `int8` quantization:

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


> [!WARNING]
> For this model, `int4` quantization may result in poor accuracy. We include it only to demonstrate the workflow.
- Weight-only `int4` quantization:

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

<!-- Tripy: EXAMPLE: IF_FP8 Start -->
3. `float8` quantization:

    ```bash
    python3 example.py --input-text "What is the answer to life, the universe, and everything?" --seed=0 --quant-mode float8
    ```
<!-- Tripy: EXAMPLE: IF_FP8 End -->
