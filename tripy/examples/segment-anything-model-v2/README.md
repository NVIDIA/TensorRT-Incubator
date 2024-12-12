# SAM2: Segment Anything in Images and Videos

## Introduction

This is an implementation of SAM2 model ([original repository](https://github.com/facebookresearch/sam2/tree/main) by Meta).

## Running The Example

1. Install prerequisites:

    ```bash
    sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 jpeginfo -y
    python3 -m pip install -r requirements.txt
    ```

2. Retrieve the images and the checkpoint:

    ```bash
    chmod +x download_images.sh
    ./download_images.sh
    mkdir checkpoints && cd checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
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
    Last frame object 2 has mask properties: volume {16338~5%}, centre (0.0, {95.80028155220957~5%}, {133.8682825315216~5%})
    Last frame object 3 has mask properties: volume {4415~5%}, centre (0.0, {161.95605889014723~5%}, {421.4523216308041~5%})
    ```
    Tripy: TEST: EXPECTED_STDOUT End
    -->


## License
The SAM2 model checkpoints and associated model code are sourced from Meta's [SAM2 repository](https://github.com/facebookresearch/sam2/tree/main) and are licensed under the Apache 2.0 license (included as LICENSE_sam2 in our sample).
