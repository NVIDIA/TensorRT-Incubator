# Quantization

```{contents} Table of Contents
:depth: 3
```

## Using Quantized Modules

Various modules predefined by Tripy support quantization. For example, the {class}`tripy.Linear` module includes two arguments to configure the quantization mode. Let's construct the following quantized linear module:

```py
# doc: print-locals quant_linear
quant_linear = tp.Linear(
    4,
    2,
    quant_dtype=tp.int8,
    weight_quant_dim=None,
)
```

As described in {class}`tripy.Linear`, the quantized linear module has 2 additional {class}`tripy.Parameter`s compared to a normal linear layer:

1. `weight_scale`: The quantization scale for `weight`.

2. `input_scale`: The quantization scale for the input.

`weight_scale` must always be provided while `input_scale` is optional. The input will be quantized only if `input_scale` is provided. For a `Linear` module in this example, only "per-tensor" quantization is allowed for the input. This is why there is no `input_quant_dim` argument.  

Let's fill the scale parameters with dummy data:

```py
# doc: print-locals quant_linear
quant_linear.weight_scale = tp.Parameter(1.0)
quant_linear.input_scale = tp.Parameter(1.0)
```

and run a forward pass to see the result:

```py
x = tp.iota((3, 4), dtype=tp.float32)
out = quant_linear(x)
```

The result still has a data type of {class}`tripy.float32`, but internally, TensorRT quantized the input and weight, executed the linear layer with {class}`tripy.int8` precision, and finally dequantized the output back to the original precision.

## Running Quantized Models

Now that we have covered how quantization works in {class}`tripy.Linear`, we will walk through the workflow of running a real-world quantized model: [nanoGPT](source:/examples/nanogpt/).

### Calibration With Model Optimizer

<!-- Tripy: IGNORE Start -->

The quantization scales are not available unless the model was trained with QAT (quantization-aware training). We need to perform another step called calibration to compute the correct scales for each quantized layer. There are many ways to do calibration, one of which is using the `nvidia-modelopt` toolkit. To install it, run:

```sh
python3 -m pip install --extra-index-url https://pypi.nvidia.com nvidia-modelopt==0.11.0 transformers datasets
```

First, let's get the pre-trained GPT model from hugging face:

```py
# doc: no-print-locals
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
```

Then, we perform int8 weight-only quantization:

```py
from transformers import AutoTokenizer
import modelopt.torch.quantization as mtq

from modelopt.torch.utils.dataset_utils import create_forward_loop

# define the modelopt quant configs
quant_cfg = mtq.INT8_DEFAULT_CFG
# disable input quantization for weight-only
# quantized linear modules
quant_cfg["quant_cfg"]["*input_quantizer"] = {
    "enable": False,
}

# define the forward loop for calibration
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

# call the api for calibration
mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
```

`mtq.quantize` replaces all linear layers specified in `quant_cfg` with `QuantLinear` layers, which contain the calibrated parameters.

### Load Scales Into The Tripy Model

Let's take a look at one of the `QuantLinear` produced by model optimizer:

```py
print(model.transformer.h[0].attn.c_attn)
```

The `amax` attribute gives us the dynamic range of the tensor. Tripy requires scaling factors, so we can convert it like so:

```py
def convert_to_scale(amax, maxbound):
    return amax.float() / maxbound
```

Let's convert the `amax` to the scaling factor and load it to a compatible {class}`tripy.Linear` module:

```py
# doc: print-locals weight_only_qlinear
weight_only_qlinear = tp.Linear(
    768,
    2304,
    quant_dtype=tp.int8,
    weight_quant_dim=0,
)
quantizer = model.transformer.h[0].attn.c_attn.weight_quantizer
scale = convert_to_scale(quantizer.export_amax(), quantizer.maxbound)
scale = scale.squeeze().contiguous()
weight_only_qlinear.weight_scale = tp.Parameter(scale)
```

For an example of how to load weights from a quantized model, refer to [load_quant_weights_from_hf](source:/examples/nanogpt/weight_loader.py) from the nanoGPT example.

<!-- Tripy: IGNORE End -->
