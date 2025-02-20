# Quantization

**Quantization** reduces memory and compute requirements by running operations in low precision:
- **Scaling** is required to translate to/from low precision.
- **Scaling factors** are chosen such that they minimize accuracy loss.
- They can be either:
    - Loaded into quantization-enabled {class}`nvtripy.Module`s, or
    - Used with {func}`nvtripy.quantize`/{func}`nvtripy.dequantize`.

:::{seealso}
The
[TensorRT developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
explains quantization in more detail.
:::


## Post-Training Quantization With ModelOpt

If the model was not trained with quantization-aware training (QAT), we can use
[TensorRT ModelOpt](https://nvidia.github.io/TensorRT-Model-Optimizer/index.html)
to do **calibration** to determine scaling factors.

:::{admonition} Info
**Calibration** runs a model with a small set of input data to determine the
numerical distribution of each tensor.

The **dynamic range** is the most important range within this distribution and
scales are chosen to target this range.
:::

Let's calibrate a GPT model:

1. Install ModelOpt:

    ```bash
    python3 -m pip install nvidia-modelopt==0.11.1 transformers==4.46.2 datasets==2.21.0
    ```

2. Download the model:

    ```py
    # doc: no-print-locals
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    ```

3. Calibrate for `int8` precision:

    1. Define the forward pass:

        ```py
        # doc: no-output
        from transformers import AutoTokenizer
        from modelopt.torch.utils.dataset_utils import create_forward_loop

        MAX_SEQ_LEN = 512
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            use_fast=True,
            model_max_length=MAX_SEQ_LEN,
            padding_side="left",
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        forward_loop = create_forward_loop(
            model=model,
            dataset_name="cnn_dailymail",
            tokenizer=tokenizer,
            device=model.device,
            num_samples=8,
        )
        ```

    2. Set up quantization configuration:

        ```py
        import modelopt.torch.quantization as mtq

        quant_cfg = mtq.INT8_DEFAULT_CFG
        ```

    3. Run calibration to replace linear layers with
        [`QuantLinear`](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.nn.modules.quant_linear.html#modelopt.torch.quantization.nn.modules.quant_linear.QuantLinear),
        which contain calibration information:

        ```py
        # doc: no-output
        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
        ```


The `amax` attributes of `QuantLinear`'s quantizers specify **dynamic ranges**:

```py
torch_qlinear = model.transformer.h[0].attn.c_attn
print(torch_qlinear)
```

We must convert dynamic ranges to scaling factors to load them into Tripy:

```py
def get_scale(quantizer):
    amax = quantizer.export_amax()
    # `maxbound` is the maximum value representible by the data type.
    # For `int8`, this is 127.
    scale = amax.float() / quantizer.maxbound
    return tp.Tensor(scale.squeeze().contiguous())

input_scale = get_scale(torch_qlinear.input_quantizer)
weight_scale = get_scale(torch_qlinear.weight_quantizer)
```


## Loading Scales Into Tripy

### Using Modules

Modules that support quantization usually:
- Expose additional model parameters for scales.
- Accept arguments that control how quantization is performed.

Let's load the scales into an {class}`nvtripy.Linear` module:

```py
qlinear = tp.Linear(
    768,
    2304,
    # The data type to quantize to:
    quant_dtype=tp.int8,
    # The dimension along which the weights are quantized:
    weight_quant_dim=torch_qlinear.weight_quantizer.axis)

# Load weights:
qlinear.weight = tp.Tensor(torch_qlinear.weight.detach().contiguous())
qlinear.bias = tp.Tensor(torch_qlinear.bias.detach().contiguous())

# Load scaling factors:
qlinear.input_scale = input_scale
qlinear.weight_scale = weight_scale
```

:::{note}
We use scales from ModelOpt here, but scaling factors can come from anywhere.
:::

We can run it just like a regular `float32` module.
Inputs/weights are quantized internally:

```py
input = tp.ones((1, 768), dtype=tp.float32)

output = qlinear(input)
```

:::{seealso}
`load_quant_weights_from_hf` in the [nanoGPT weight loader](source:/examples/nanogpt/weight_loader.py)
is an example of loading scaling factors for an entire model.
:::


### Manually

When using {func}`nvtripy.quantize`/{func}`nvtripy.dequantize`,
`dequantize` must **immediately follow** `quantize`.

TensorRT will **rotate** `dequantize` over subsequent ops as needed.

:::{seealso}
The
[TensorRT developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#qdq-placement-recs)
includes recommendations on placement of quantization and dequantization ops.
:::

<!-- We cannot print the quantized input/weight below since that would break Q/DQ rotation -->

To mimic the behavior of the {class}`nvtripy.Linear` module above, we can:

1. Quantize the input:

    ```py
    # doc: no-print-locals
    input = tp.ones((1, 768), dtype=tp.float32)

    input = tp.quantize(input, input_scale, dtype=tp.int8)
    # Note the placement of dequantize:
    input = tp.dequantize(input, input_scale, dtype=tp.float32)
    ```

2. Quantize the weights:

    ```py
    # doc: no-print-locals
    weight = tp.Tensor(torch_qlinear.weight.detach().contiguous())

    dim = torch_qlinear.weight_quantizer.axis
    weight = tp.quantize(weight, weight_scale, dtype=tp.int8, dim=dim)
    weight = tp.dequantize(weight, weight_scale, dtype=tp.float32, dim=dim)
    ```

3. Perform the computation (matrix multiply in this case):

    ```py
    # doc: no-print-locals bias
    bias = tp.Tensor(torch_qlinear.bias.detach().contiguous())

    output = input @ tp.transpose(weight, 0, 1) + bias
    ```

:::{warning}
**Evaluating** the tensor produced by `dequantize` will affect accuracy.

- **Why:** Evaluation replaces the tensor with a constant, losing information
    like which op produced it.

    So, TensorRT won't see `dequantize` when evaluating subsequent ops and
    won't **rotate** it correctly.

For example, **don't** do this:
```py
# doc: no-eval
tensor = tp.ones(...)

tensor = tp.quantize(tensor, ...)
tensor = tp.dequantize(tensor, ...)

# The `print` below will trigger an evaluation of the tensor which will prevent
# TensorRT from rotating the dequantization node. This will affect accuracy!
print(tensor)

# Rest of the program, including some computation involving tensor
...
```
:::
