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
import os
import time
import requests
from pathlib import Path
from PIL import Image
import io


def verify_jpeg(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except:
        return False


def download_file(url, filepath, max_retries=5, timeout=5):
    """Download file with retries and timeout."""
    for attempt in range(max_retries):
        try:
            print(f"Downloading {filepath}...")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)

            # Verify if it's a valid JPEG
            if verify_jpeg(filepath):
                return True
            else:
                print(f"Invalid JPEG file {filepath}")
                os.remove(filepath)  # Remove invalid file

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Error downloading {filepath} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(0.01)
            continue
    print(f"Failed to download {filepath} after {max_retries} attempts")
    return False


def main():
    # Download test image for image segmentation
    truck_url = "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg"
    download_file(truck_url, "truck.jpg")

    # Create bedroom directory if it doesn't exist
    bedroom_dir = Path("bedroom")
    bedroom_dir.mkdir(exist_ok=True)

    # Download images for video segmentation
    base_url = "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/videos/bedroom/{:05d}.jpg"

    for i in range(200):  # 0 to 199
        filepath = bedroom_dir / f"{i:05d}.jpg"
        download_file(base_url.format(i), filepath)


if __name__ == "__main__":
    main()
