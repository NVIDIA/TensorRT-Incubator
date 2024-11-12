# SAM2: Segment Anything in Images and Videos

## Introduction

This is an implementation of SAM2 model ([original repository](https://github.com/facebookresearch/sam2/tree/main) by Meta).

## Running The Example

### Image pipeline

1. Install prerequisites:

    ```bash
    apt-get install ffmpeg libsm6 libxext6  -y
    mkdir checkpoints && cd checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    python3 -m pip install -r requirements.txt
    ```

2. Run the example:

    ```bash
    python3 image_demo.py
    ```

### Video segmentation pipeline

TBD
