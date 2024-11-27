# SAM2: Segment Anything in Images and Videos

## Introduction

This is an implementation of SAM2 model ([original repository](https://github.com/facebookresearch/sam2/tree/main) by Meta).

## Running The Example

### Image pipeline

1. Install prerequisites:

    ```bash
    sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
    wget -O truck.jpg https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg
    mkdir checkpoints && cd checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    python3 -m pip install -r requirements.txt
    ```

2. Run the example:

    ```bash
    python3 image_demo.py
    ```

    <!-- Tripy: TEST: EXPECTED_STDOUT Start -->
    <!--
    ```
    Generating image embedding took {137.81~10%} ms per run (averaged over 100 runs, with 5 warmup runs)
    Predicting masks took {37.78~10%} ms per run (averaged over 100 runs, with 5 warmup runs)
    Scores for each prediction: {0.78759766~5%} {0.640625~5%} {0.05099487~5%}
    ```
     -->
    <!-- Tripy: TEST: EXPECTED_STDOUT End -->

### Video segmentation pipeline

TBD


## License
The SAM2 model checkpoints and associated model code are sourced from Meta's [SAM2 repository](https://github.com/facebookresearch/sam2/tree/main) and are licensed under the Apache 2.0 license (included as LICENSE_sam2 in our sample).
