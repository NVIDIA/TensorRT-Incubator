#!/bin/bash

# Download test image for image segmentation
wget -q -O truck.jpg https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg

# Create bedroom directory if it doesn't exist
mkdir -p bedroom

# Download and verify images from 00000 to 00199
for i in $(seq -f "%05g" 0 199); do
    # Loop until file exists and is a valid JPEG
    while [ ! -f "bedroom/${i}.jpg" ] || ! jpeginfo -c "bedroom/${i}.jpg" >/dev/null 2>&1; do
        wget -q -O "bedroom/${i}.jpg" \
            --retry-connrefused \
            --tries=0 \
            --timeout=5 \
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/videos/bedroom/${i}.jpg"
        sleep 0.01
    done
done
