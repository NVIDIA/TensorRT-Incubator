# DLRMv3 HSTU with Tripy

[Paper](https://arxiv.org/pdf/1906.00091) | [Notes](https://docs.google.com/document/d/1GQtAcGmLpMvt-TCQwxYEK7NQQR9jw-dh6le_DyZGlVc/edit?usp=sharing)

## Navigating Models and Test Files

__Original Torch Model Implementation:__
1. We've provided a copy of `generative_recommenders/research/modeling/sequential/hstu.py` with additional debugging information added, and its dependencies.
2. An implementation that does not use jagged tensors is available at `generative_recommenders/research/modeling/sequential/hstu_dense.py`.
3. You may want to clone [original repo](https://github.com/meta-recsys/generative-recommenders/tree/main) and add in the new files.

__Tripy Implementation:__

All files and modules are prefixed with `tripy_` and `Tripy` respectively. The following have been converted to Tripy:
```
generative_recommenders/
generative_recommenders/research/modeling/sequential/tripy_embedding_modules.py
generative_recommenders/research/modeling/sequential/tripy_hstu_dense.py
generative_recommenders/research/modeling/sequential/tripy_input_features_preprocessors.py
generative_recommenders/research/modeling/sequential/tripy_output_postprocessors.py
generative_recommenders/research/modeling/sequential/tripy_utils.py
generative_recommenders/research/modeling/tripy_initialization.py
generative_recommenders/research/modeling/tripy_similarity_module.py
```

## Testing the Models

1. Launch container `docker run --pull always --gpus all -it -v $(pwd):/tripy/ --rm ghcr.io/nvidia/tensorrt-incubator/tripy`
2. Install dependencies `pip3 install -r requirements.txt`

The inference code follows the config used for the public experiments portion of the paper with research model.

3. Run the torch model and compare with dense-tensor-only version:
`python3 test_research_pipeline.py`

4. Run the tripy model and compare it with torch:
`python3 tripy_hstu_example.py`

If you wish to run the MLPerf benchmark, you will need to follow the instructions in the README of the original repo. This uses a different implementation of the DLRMv3 that uses many custom OAI Triton kernels.

## Future Work Roadmap

1. Compare potential divergence between research model and production inference model (possible that relative attention bias is applied differently, for example)
2. Low-hanging fruit for Tripy optimizations from the raw translation (INormalization / skip useless affine transform, constant folding, IAttention) & check casting/precisions for potential accuracy issues

The rest would be expected to be driven by MLPerf profiling & optimization goals
3. Enable support for ragged tensors
4. Enable KVCache support
5. etc.

