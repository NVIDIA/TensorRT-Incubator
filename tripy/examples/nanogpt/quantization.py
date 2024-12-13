#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import time

from transformers import AutoTokenizer
import modelopt.torch.quantization as mtq

from modelopt.torch.utils.dataset_utils import create_forward_loop


def modelopt_quantize(model_hf, quant_mode):
    # quantize and calibrate pytorch model using modelopt
    start_time = time.perf_counter()
    MAX_SEQ_LEN = 2048
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        model_max_length=MAX_SEQ_LEN,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if quant_mode == "int8-weight-only":
        quant_cfg = mtq.INT8_DEFAULT_CFG
        quant_cfg["quant_cfg"]["*input_quantizer"] = {
            "enable": False,
        }
    elif quant_mode == "int4-weight-only":
        quant_cfg = mtq.INT4_AWQ_CFG
    elif quant_mode == "float8":
        quant_cfg = mtq.FP8_DEFAULT_CFG
    else:
        raise NotImplementedError(f"Unsupported quantization mode: {quant_mode}")

    calib_size = 64
    batch_size = 1
    if quant_mode == "int4-weight-only":
        calib_size = 16
        batch_size = 16
    forward_loop = create_forward_loop(
        model=model_hf,
        dataset_name="cnn_dailymail",
        tokenizer=tokenizer,
        device=model_hf.device,
        batch_size=batch_size,
        num_samples=calib_size,
    )

    mtq.quantize(model_hf, quant_cfg, forward_loop=forward_loop)
    end_time = time.perf_counter()
    print(f"Quantization took {end_time - start_time} seconds.")
    return model_hf
