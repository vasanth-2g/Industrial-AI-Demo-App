# DINOv2 Vision Flow Architecture

This file explains the current vision flow used by the app.

## Summary

The app does not train DINOv2.

DINOv2 is used as a frozen visual feature extractor. The app compares uploaded
image patches against a saved normal/reference patch memory and marks regions
that look unusual.

```text
uploaded image
-> optional robot/asset ROI crop
-> frozen DINOv2 feature extraction
-> compare patches with normal patch memory
-> anomaly score
-> defect region mask
-> transparent colored overlay
-> RCA uses vision output + telemetry output + RAG output
```

## What Is Trained

Nothing in DINOv2 is trained during app runtime.

```text
DINOv2 model training: no
DINOv2 fine-tuning: no
Runtime classifier training: no
```

The model files are loaded from:

```text
app/backend/vision_dinov2/facebook_dinov2_base
```

The loading code is in:

```text
app/backend/vision_runtime.py
```

## What Is Saved

The app uses a saved normal patch memory:

```text
app/backend/vision_dinov2/normal_patch_memory.pt
```

This file stores DINOv2 patch embeddings from normal/reference images. It is
used as the baseline for anomaly detection.

```text
new image patch
-> DINOv2 embedding
-> compare with normal patch embeddings
-> high distance means unusual region
```

This is not DINOv2 training. It is a reference memory bank.

## Current Runtime Flow

1. Frontend uploads:

```text
input.txt
inspection image
```

2. Backend saves the image into:

```text
app/runtime/runs/<timestamp>/inputs/
```

3. `vision_runtime.py` loads the image.

4. DINOv2 converts image patches into embeddings.

5. Each patch is compared with `normal_patch_memory.pt`.

6. The app calculates:

```text
anomaly_score
is_anomaly
pixel_threshold
predicted_fault
defect overlay image
```

7. The result is returned to the frontend as:

```json
{
  "anomaly_score": 0.79,
  "is_anomaly": true,
  "predicted_fault": "oil_leak",
  "heatmap_url": "/api/runtime/runs/.../outputs/vision_dinov2_result.png"
}
```

## Fault Type Labels

The app does not currently use a trained image classifier for fault type.

Fault names are inferred from image path or context words:

```text
crack
corrosion
oil_leak
wear
overheating
```

Example:

```text
oil_leak.png -> predicted_fault = oil_leak
crack_surface.png -> predicted_fault = crack
```

## Overlay Colors

The output image uses transparent color marking by fault type:

```text
oil_leak    -> blue
crack       -> red
corrosion   -> orange
wear        -> purple
overheating -> orange/red
unknown     -> teal
```

The image does not show score text or region labels. It only marks the detected
region.

## Problem With Busy Plant Images

Plant images may contain:

```text
pipes
windows
walls
metal frames
background machines
lighting changes
```

DINOv2 anomaly patches can fire on these background areas. That is why some
defect markings may appear outside the robot.

## Improved Flow: Robot ROI Before DINOv2

The proposed improved flow is:

```text
uploaded image
-> object detection finds robot / inspected asset
-> SAM segmentation creates robot mask
-> remove background
-> run DINOv2 only on robot crop
-> generate defect overlay on crop
-> send output to frontend
```

Notebook for testing this:

```text
app/robot_roi_before_dinov2.ipynb
```

The notebook supports:

```text
manual bounding box crop
automatic GrabCut foreground crop
open-vocabulary object detection + SAM segmentation
DINOv2 anomaly detection on cropped image
```

Preferred order:

```text
1. Manual box if provided
2. Object detection + SAM robot-only crop
3. GrabCut foreground crop fallback
```

For production, object detection + SAM is the best direction because DINOv2
should only inspect the robot/asset, not pipes, windows, walls, or background
plant structure.

## Runtime App Flow

The backend inference path now attempts the same ROI flow as the notebook:
OWL-ViT object detection first, SAM segmentation second, then DINOv2 scoring
on the background-removed crop. If OWL-ViT/SAM are unavailable, it falls back
to OpenCV foreground cropping.

```text
POST /api/infer
-> save uploaded image
-> OWL-ViT detects robot/mechanical asset
-> SAM segments the asset mask
-> remove background and save robot-only crop
-> DINOv2 patch scoring on processed crop
-> return processed_image_path, roi_mode, roi_box, anomaly score, overlay image
```

If OWL-ViT/SAM cannot detect a robot or mechanical asset, the scenario is
rejected before telemetry, RAG, or RCA classification continues.

```text
non-mechanical image
-> no robot/mechanical asset detected
-> return HTTP 422
-> error = image_rejected
-> do not run telemetry/RAG/RCA
```

The crop metadata appears in:

```text
vision.evidence.roi_mode
vision.evidence.roi_box
vision.processed_image_path
```

Expected preferred ROI mode:

```text
owlvit_sam_roi_crop
```

Fallback mode:

```text
foreground_roi_crop
```

## Updating Normal Patch Memory

The app can rebuild DINOv2 normal memory from uploaded clean/reference images.

```text
GET  /api/vision/memory
POST /api/vision/memory
```

The RAG page has a **DINOv2 Normal Reference Memory** panel. Upload clean
normal images of the same robot/machine there. The backend converts those
images into DINOv2 patch embeddings and replaces:

```text
app/backend/vision_dinov2/normal_patch_memory.pt
```

This still does not train DINOv2. It only updates the normal/reference memory
used for anomaly comparison.

## Backend Files

Main vision code:

```text
app/backend/vision_runtime.py
```

Model config:

```text
app/model_config.json
```

Vision model folder:

```text
app/backend/vision_dinov2
```

Runtime output folder:

```text
app/runtime/runs/<timestamp>/outputs/
```

## Future Upgrade Options

### Option 1: Rebuild Normal Patch Memory

Use real normal robot images to rebuild:

```text
normal_patch_memory.pt
```

This keeps DINOv2 frozen but improves anomaly comparison for the robot domain.

### Option 2: Add Robot Segmentation

Add a detector + segmenter so only the robot is analyzed.

Recommended architecture:

```text
Camera/Image
-> Object Detection (YOLO, open-vocabulary detector, or asset detector)
-> Segmentation Mask (SAM)
-> Background Removal
-> Robot-only image
-> DINOv2 anomaly localization
```

This is the best fix for background false positives.

### Option 3: Add Fault Classifier

Train or add a classifier for:

```text
oil leak
crack
corrosion
wear
overheating
normal
```

Then fault labels would come from the model, not from filenames.

## Current Status

```text
DINOv2 frozen feature extractor: done
normal patch memory anomaly scoring: done
transparent defect overlay: done
robot ROI crop notebook: done
object detection + SAM notebook flow: done
backend detector/SAM integration: not yet
trained fault classifier: not yet
```
