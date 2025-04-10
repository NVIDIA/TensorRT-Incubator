# SAM2: Segment Anything in Images and Videos

## Introduction

This is an implementation of SAM2 model ([original repository](https://github.com/facebookresearch/sam2/tree/main) by Meta).

## Running The Example

1. Install prerequisites:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2. Retrieve the images and the checkpoint:

    ```bash
    python3 download_test_data.py
    sh checkpoints/download_ckpt.sh
    ```

### Image pipeline

1. Run the example:

    ```bash
    python3 image_demo.py
    ```

    <!--
    Tripy: TEST: EXPECTED_STDOUT Start
    ```
    Scores for each prediction: {0.78759766~5%} {0.640625~5%} {0.05099487~5%}
    ```
    Tripy: TEST: EXPECTED_STDOUT End
    -->

### Video segmentation pipeline

1. Run the example:

    ```bash
    python3 video_demo.py
    ```

    <!--
    Tripy: TEST: EXPECTED_STDOUT Start
    ```
    Frame 180, object 2 has mask properties: volume {16340~5%}, centre (0.0, {96~5%}, {134~5%})
    Frame 180, object 3 has mask properties: volume {4414~5%}, centre (0.0, {162~5%}, {421~5%})
    ```
    Tripy: TEST: EXPECTED_STDOUT End
    -->


## License
The SAM2 model checkpoints and associated model code are sourced from Meta's [SAM2 repository](https://github.com/facebookresearch/sam2/tree/main) and are licensed under the Apache 2.0 license (included as LICENSE_sam2 in our sample).
