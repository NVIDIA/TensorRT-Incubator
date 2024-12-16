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
import gc
import os
import cv2
import torch
from typing import Optional
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def main(video_dir: str, text: str, save_path: Optional[str] = None):
    """
    Main execution function.

    Args:
        video_path (str): Path to where video frames are stored
        save_path (str, optional): Directory to save visualizations

    Returns:
        Dict[str, np.ndarray]: Processing results
    """

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cuda"))

    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # scan all the JPEG frame names in this directory
    frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=video_dir)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    """
    Prompt Grounding DINO and SAM image predictor to get the box and mask
    """

    # prompt grounding dino to get the box coordinates on specific frame
    img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    image = Image.open(img_path)

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.3, target_sizes=[image.size[::-1]]
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image_batch([np.array(image.convert("RGB"))])

    # process the detection results
    input_boxes = results[0]["boxes"]
    OBJECTS = results[0]["labels"]

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor._predict(
        point_coords=None,
        point_labels=None,
        boxes=input_boxes,
        multimask_output=True,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks[:, 0, :, :]

    """
    Register each object's positive points to video predictor
    """
    input_boxes = input_boxes.cpu().numpy()
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )

    """
    Propagate the video predictor to get the segmentation results for each frame
    """
    torch.cuda.empty_cache()
    gc.collect()

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
        }

    """
    Visualize the segment results across the video and save them
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))

        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks,  # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids]
        )
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(save_path, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)


if __name__ == "__main__":
    main("./bedroom", "boy.girl.", save_path="output")
