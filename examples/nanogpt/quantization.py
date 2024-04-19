from transformers import AutoTokenizer
import ammo.torch.quantization as atq

from ammo.torch.utils.dataset_utils import create_forward_loop


def ammo_quantize(model_hf, quant_mode):
    # quantize and calibrate pytorch model using ammo

    MAX_SEQ_LEN = 2048
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        model_max_length=MAX_SEQ_LEN,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if quant_mode == "int8-weight-only":
        quant_cfg = atq.INT8_DEFAULT_CFG
        quant_cfg["quant_cfg"]["*input_quantizer"] = {
            "enable": False,
        }
    elif quant_mode == "fp8":
        quant_cfg = atq.FP8_DEFAULT_CFG
    elif quant_mode == "int4-weight-only":
        quant_cfg = atq.INT4_AWQ_CFG
    else:
        raise NotImplementedError(f"Unsupported quantization mode: {quant_mode}")

    calib_size = 512
    batch_size = 1
    if quant_mode == "int4-weight-only":
        calib_size = 32
        batch_size = 16
    forward_loop = create_forward_loop(
        model=model_hf,
        dataset_name="cnn_dailymail",
        tokenizer=tokenizer,
        device=model_hf.device,
        batch_size=batch_size,
        num_samples=calib_size,
    )

    atq.quantize(model_hf, quant_cfg, forward_loop=forward_loop)
    print("Quantization complete.")
    return model_hf
