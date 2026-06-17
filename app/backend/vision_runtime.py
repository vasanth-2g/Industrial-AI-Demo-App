from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from model_config import config_value, resolve_app_path

DEFAULT_VISION_MODEL_DIR = Path(__file__).resolve().parent / "vision_dinov2"
VISION_MODEL_DIR = resolve_app_path(config_value("vision", "model_dir"), DEFAULT_VISION_MODEL_DIR)
DINO_LOCAL_DIR = resolve_app_path(
    config_value("vision", "dinov2_local_dir"),
    VISION_MODEL_DIR / "facebook_dinov2_base",
)
OWLVIT_LOCAL_DIR = VISION_MODEL_DIR / "google_owlvit_base_patch32"
SAM_LOCAL_DIR = VISION_MODEL_DIR / "facebook_sam_vit_base"
SIMILARITY_CHUNK_SIZE = 5000
TOP_PATCH_FRACTION = 0.05
_DINO_CACHE: dict[str, Any] = {}
_ROI_CACHE: dict[str, Any] = {}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
ASSET_DETECTION_LABELS = [
    "industrial robot",
    "robot arm",
    "robotic arm",
    "inspection robot",
    "mechanical arm",
    "machine part",
    "industrial machine",
    "mechanical component",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def default_demo_package(repo_root: Path) -> Path:
    return (repo_root / "app").resolve()


def vision_metadata() -> dict[str, Any]:
    return load_json(VISION_MODEL_DIR / str(config_value("vision", "metadata_file", "metadata.json")))


def load_dinov2_runtime() -> dict[str, Any]:
    if _DINO_CACHE:
        return _DINO_CACHE

    memory_path = VISION_MODEL_DIR / str(config_value("vision", "normal_patch_memory_file", "normal_patch_memory.pt"))
    if not memory_path.is_file():
        raise FileNotFoundError(memory_path)

    state = torch.load(memory_path, map_location="cpu")
    model_name = state.get("model_name", "facebook/dinov2-base")
    model_source = DINO_LOCAL_DIR if DINO_LOCAL_DIR.is_dir() else model_name
    image_size = int(state.get("image_size", 224))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(
        model_source,
        size={"height": image_size, "width": image_size},
        crop_size={"height": image_size, "width": image_size},
        use_fast=False,
        local_files_only=DINO_LOCAL_DIR.is_dir(),
    )
    model = AutoModel.from_pretrained(
        model_source,
        local_files_only=DINO_LOCAL_DIR.is_dir(),
    ).to(device)
    model.eval()

    _DINO_CACHE.update(
        {
            "memory_bank": state["memory_bank"].float().contiguous(),
            "model_name": model_name,
            "model_source": str(model_source),
            "image_size": image_size,
            "grid_height": int(state.get("grid_height", image_size // int(state.get("patch_size", 14)))),
            "grid_width": int(state.get("grid_width", image_size // int(state.get("patch_size", 14)))),
            "normal_memory_sources": state.get("normal_memory_sources", []),
            "evaluation_mode": state.get("evaluation_mode", "functional_demo_no_independent_normal_test"),
            "processor": processor,
            "model": model,
            "device": device,
        }
    )
    return _DINO_CACHE


def infer_fault_from_path(image_path: str | None, fallback: str = "visual_anomaly") -> str:
    if not image_path:
        return fallback

    lowered = image_path.lower()
    if any(token in lowered for token in ("crack", "fracture", "split")):
        return "crack"
    if any(token in lowered for token in ("corrosion", "corossion", "corosion", "rust", "oxidation")):
        return "corrosion"
    if any(token in lowered for token in ("oil", "leak", "lubricant", "grease")):
        return "oil_leak"
    if any(token in lowered for token in ("wear", "abrasion", "scratch", "scoring")):
        return "wear"
    if any(token in lowered for token in ("heat", "overheat", "overheating", "burn", "thermal")):
        return "overheating"
    return fallback


def heatmap_path_for(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "vision_dinov2_result.png"


def roi_path_for(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "vision_robot_roi.png"


def extract_patch_embeddings(image: Image.Image, runtime: dict[str, Any]) -> torch.Tensor:
    inputs = runtime["processor"](images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(runtime["device"])
    with torch.inference_mode():
        outputs = runtime["model"](pixel_values=pixel_values)
        patches = F.normalize(outputs.last_hidden_state[:, 1:, :].float(), dim=-1)
    return patches[0].cpu()


def torch_device_info() -> tuple[str, int, str]:
    if torch.cuda.is_available():
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        return "cuda", 0, backend
    return "cpu", -1, "cpu"


def pad_box(box: list[float] | tuple[float, float, float, float], image_shape: tuple[int, ...], pad_ratio: float = 0.08) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    pad = int(max(x2 - x1, y2 - y1) * pad_ratio)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def crop_from_segmentation_mask(rgb: np.ndarray, mask: np.ndarray, fallback_box: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        x1, y1, x2, y2 = fallback_box
        return rgb[y1:y2, x1:x2], fallback_box

    ys, xs = np.where(binary > 0)
    box = pad_box((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)), rgb.shape, 0.04)
    x1, y1, x2, y2 = box
    robot_only = rgb.copy()
    robot_only[binary == 0] = 255
    return robot_only[y1:y2, x1:x2], box


def load_roi_runtime() -> dict[str, Any]:
    if _ROI_CACHE:
        return _ROI_CACHE

    try:
        from transformers import SamModel, SamProcessor, pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"OWL-ViT/SAM runtime is unavailable: {type(exc).__name__}: {exc}") from exc

    device, pipeline_device, backend = torch_device_info()
    detector_source = str(OWLVIT_LOCAL_DIR) if OWLVIT_LOCAL_DIR.is_dir() else "google/owlvit-base-patch32"
    sam_source = str(SAM_LOCAL_DIR) if SAM_LOCAL_DIR.is_dir() else "facebook/sam-vit-base"
    detector = pipeline(
        task="zero-shot-object-detection",
        model=detector_source,
        device=pipeline_device,
    )
    sam_processor = SamProcessor.from_pretrained(sam_source, use_fast=False)
    sam_model = SamModel.from_pretrained(sam_source).to(device)
    sam_model.eval()

    _ROI_CACHE.update(
        {
            "detector": detector,
            "sam_processor": sam_processor,
            "sam_model": sam_model,
            "device": device,
            "backend": backend,
            "detector_source": detector_source,
            "sam_source": sam_source,
        }
    )
    return _ROI_CACHE


def crop_asset_region_with_owlvit_sam(image_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    runtime = load_roi_runtime()
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    detections = runtime["detector"](image, candidate_labels=ASSET_DETECTION_LABELS)
    detections = sorted(detections, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    detections = [item for item in detections if float(item.get("score", 0.0)) >= 0.05]
    if not detections:
        raise RuntimeError("OWL-ViT did not detect a robot or mechanical asset.")

    best = detections[0]
    box_dict = best["box"]
    det_box = pad_box(
        [box_dict["xmin"], box_dict["ymin"], box_dict["xmax"], box_dict["ymax"]],
        rgb.shape,
        0.10,
    )
    processor = runtime["sam_processor"]
    model = runtime["sam_model"]
    inputs = processor(image, input_boxes=[[list(det_box)]], return_tensors="pt").to(runtime["device"])
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores.cpu()[0, 0]
    best_mask_index = int(torch.argmax(scores).item())
    mask = masks[0, best_mask_index].numpy().astype(np.uint8)
    crop, crop_box = crop_from_segmentation_mask(rgb, mask, det_box)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(output_path)
        crop_path = output_path
    else:
        crop_path = image_path

    return {
        "path": str(crop_path),
        "mode": "owlvit_sam_roi_crop",
        "box": [int(value) for value in crop_box],
        "error": "",
        "detected_label": best.get("label"),
        "detector_score": float(best.get("score", 0.0)),
        "gpu_backend": runtime["backend"],
        "detector_source": runtime["detector_source"],
        "sam_source": runtime["sam_source"],
    }


def crop_asset_region(image_path: Path, output_path: Path | None = None, require_semantic_asset: bool = False) -> dict[str, Any]:
    try:
        return crop_asset_region_with_owlvit_sam(image_path, output_path)
    except Exception as exc:
        roi_error = f"{type(exc).__name__}: {exc}"
        if require_semantic_asset:
            return {
                "path": str(image_path),
                "mode": "rejected_no_mechanical_asset",
                "box": None,
                "error": (
                    "No robot or mechanical asset was detected by OWL-ViT/SAM. "
                    f"Scenario was not processed. Details: {roi_error}"
                ),
            }

    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "path": str(image_path),
            "mode": "original_unreadable",
            "box": None,
            "error": f"{roi_error}; OpenCV could not read image.",
        }

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = foreground_object_mask(rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "path": str(image_path),
            "mode": "original_no_roi",
            "box": None,
            "error": f"{roi_error}; no foreground object contour found.",
        }

    height, width = rgb.shape[:2]
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < max(24, height * width * 0.01):
        return {
            "path": str(image_path),
            "mode": "original_small_roi",
            "box": None,
            "error": f"{roi_error}; detected foreground object was too small.",
        }

    x, y, w, h = cv2.boundingRect(largest)
    pad = int(max(w, h) * 0.08)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    cleaned = rgb.copy()
    cleaned[mask == 0] = 255
    crop = cleaned[y1:y2, x1:x2]

    if output_path is None:
        output_path = image_path
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(output_path)

    return {
        "path": str(output_path),
        "mode": "foreground_roi_crop",
        "box": [int(x1), int(y1), int(x2), int(y2)],
        "error": f"OWL-ViT/SAM fallback: {roi_error}",
    }


def nearest_memory_distance(query_patches: torch.Tensor, runtime: dict[str, Any]) -> torch.Tensor:
    device = runtime["device"]
    memory_bank = runtime["memory_bank"]
    query_patches = query_patches.to(device)
    best_similarity = torch.full((len(query_patches),), -1.0, device=device)
    for start in range(0, len(memory_bank), SIMILARITY_CHUNK_SIZE):
        memory_chunk = memory_bank[start:start + SIMILARITY_CHUNK_SIZE].to(device)
        similarity = query_patches @ memory_chunk.T
        best_similarity = torch.maximum(best_similarity, similarity.max(dim=1).values)
    return (1.0 - best_similarity).clamp(min=0).cpu()


def score_from_patches(patch_scores: torch.Tensor) -> float:
    top_count = max(1, int(math.ceil(len(patch_scores) * TOP_PATCH_FRACTION)))
    return float(torch.topk(patch_scores, top_count).values.mean())


def dinov2_global_embedding(image_path: Path) -> list[float]:
    runtime = load_dinov2_runtime()
    image = Image.open(image_path).convert("RGB")
    patches = extract_patch_embeddings(image, runtime)
    embedding = F.normalize(patches.mean(dim=0), dim=0)
    return embedding.numpy().astype(float).tolist()


def upsample_heatmap(patch_map: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(patch_map)[None, None].float()
    return F.interpolate(
        tensor,
        size=(image_size[1], image_size[0]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()


def normalized_heatmap(heatmap: np.ndarray) -> np.ndarray:
    high = float(np.percentile(heatmap, 99))
    low = float(np.min(heatmap))
    denom = max(high - low, 1e-9)
    return np.clip((heatmap - low) / denom, 0.0, 1.0)


def titled_panel(image: Image.Image, title: str, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height + 28), "white")
    panel.paste(image.resize((width, height), Image.Resampling.LANCZOS), (0, 28))
    cv_panel = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    cv2.putText(cv_panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(cv_panel, cv2.COLOR_BGR2RGB))


def fault_overlay_style(predicted_fault: str) -> dict[str, Any]:
    styles = {
        "oil_leak": {"rgb": (37, 99, 235), "label": "Oil leak region"},
        "crack": {"rgb": (239, 68, 68), "label": "Crack indication"},
        "corrosion": {"rgb": (245, 158, 11), "label": "Corrosion region"},
        "wear": {"rgb": (168, 85, 247), "label": "Wear region"},
        "overheating": {"rgb": (249, 115, 22), "label": "Overheating region"},
    }
    return styles.get(predicted_fault, {"rgb": (20, 184, 166), "label": "Anomaly region"})


def foreground_object_mask(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    rect = (
        max(1, int(width * 0.12)),
        max(1, int(height * 0.03)),
        max(2, int(width * 0.76)),
        max(2, int(height * 0.94)),
    )
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            mask,
            rect,
            bgd_model,
            fgd_model,
            3,
            cv2.GC_INIT_WITH_RECT,
        )
        foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        foreground = np.zeros((height, width), np.uint8)
        x, y, w, h = rect
        foreground[y:y + h, x:x + w] = 255

    if int(np.count_nonzero(foreground)) < int(height * width * 0.02):
        x, y, w, h = rect
        foreground[y:y + h, x:x + w] = 255

    kernel = np.ones((11, 11), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    return cv2.dilate(foreground, kernel, iterations=1)


def refine_predicted_mask(heat_norm: np.ndarray, predicted_mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    cutoff = max(0.55, float(np.percentile(heat_norm, 86)))
    mask = ((heat_norm >= cutoff).astype(np.uint8) * 255)
    if int(mask.sum()) == 0:
        cutoff = float(np.percentile(heat_norm, 97))
        mask = ((heat_norm >= cutoff).astype(np.uint8) * 255)

    foreground = foreground_object_mask(rgb)
    focused = cv2.bitwise_and(mask, foreground)
    if int(focused.sum()) > 0:
        mask = focused

    kernel_size = max(3, int(round(min(mask.shape[:2]) * 0.018)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    min_area = max(12, int(mask.shape[0] * mask.shape[1] * 0.0005))
    clean = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, thickness=-1)
    if int(clean.sum()) == 0:
        clean = mask
    expand_kernel = np.ones((7, 7), np.uint8)
    return cv2.dilate(clean, expand_kernel, iterations=1)


def save_dinov2_result_figure(
    image: Image.Image,
    heatmap: np.ndarray,
    predicted_mask: np.ndarray,
    score: float,
    predicted_fault: str,
    output_path: Path,
) -> None:
    heat_norm = normalized_heatmap(heatmap)
    rgb = np.asarray(image.convert("RGB")).copy()
    mask = refine_predicted_mask(heat_norm, predicted_mask, rgb)
    style = fault_overlay_style(predicted_fault)
    color = tuple(int(channel) for channel in style["rgb"])

    overlay = rgb.copy()
    overlay[mask > 0] = color
    marked = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(marked, contours, -1, color, thickness=5, lineType=cv2.LINE_AA)
    canvas = Image.fromarray(marked)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def score_image(image_path: Path, output_path: Path | None, predicted_fault: str) -> dict[str, Any]:
    metadata = vision_metadata()
    runtime = load_dinov2_runtime()
    image = Image.open(image_path).convert("RGB")
    patches = extract_patch_embeddings(image, runtime)
    patch_scores = nearest_memory_distance(patches, runtime)
    patch_map = patch_scores.reshape(runtime["grid_height"], runtime["grid_width"]).numpy()
    heatmap = upsample_heatmap(patch_map, image.size)
    score = score_from_patches(patch_scores)
    threshold = float(metadata.get("image_threshold", 0.008287952281534672))
    pixel_threshold = float(metadata.get("pixel_threshold", 0.057987332344055176))
    predicted_mask = (heatmap >= pixel_threshold).astype(np.uint8)

    if output_path is not None:
        save_dinov2_result_figure(
            image,
            heatmap,
            predicted_mask,
            score,
            predicted_fault,
            output_path,
        )

    return {
        "anomaly_score": score,
        "is_anomaly": bool(score >= threshold),
        "threshold": threshold,
        "pixel_threshold": pixel_threshold,
        "image_size": [int(image.width), int(image.height)],
        "model": runtime["model_name"],
        "model_source": runtime["model_source"],
        "normal_memory_sources": runtime["normal_memory_sources"],
        "evaluation_mode": runtime["evaluation_mode"],
        "patch_grid": [runtime["grid_height"], runtime["grid_width"]],
    }


def rebuild_normal_patch_memory(image_paths: list[Path], source_label: str = "uploaded_reference") -> dict[str, Any]:
    valid_paths = [path for path in image_paths if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()]
    if not valid_paths:
        raise ValueError("No readable reference images were provided.")

    runtime = load_dinov2_runtime()
    patch_batches = []
    source_names = []
    for path in valid_paths:
        image = Image.open(path).convert("RGB")
        patches = extract_patch_embeddings(image, runtime)
        patch_batches.append(patches)
        source_names.append(path.name)

    memory_bank = torch.cat(patch_batches, dim=0).float().contiguous()
    memory_path = VISION_MODEL_DIR / str(config_value("vision", "normal_patch_memory_file", "normal_patch_memory.pt"))
    state = {
        "memory_bank": memory_bank,
        "model_name": runtime["model_name"],
        "image_size": runtime["image_size"],
        "grid_height": runtime["grid_height"],
        "grid_width": runtime["grid_width"],
        "patch_size": max(1, runtime["image_size"] // max(1, runtime["grid_height"])),
        "normal_memory_sources": source_names,
        "evaluation_mode": f"{source_label}_normal_reference_memory",
    }
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, memory_path)

    metadata_path = VISION_MODEL_DIR / str(config_value("vision", "metadata_file", "metadata.json"))
    metadata = vision_metadata()
    metadata.update(
        {
            "normal_memory_sources": source_names,
            "memory_bank_patches": int(memory_bank.shape[0]),
            "evaluation_mode": state["evaluation_mode"],
            "limitations": [
                "Normal patch memory was rebuilt from uploaded reference images.",
                "DINOv2 remains frozen; only the reference memory bank changed.",
                "Use clean/normal asset images for best anomaly localization.",
            ],
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _DINO_CACHE.clear()
    return {
        "status": "updated",
        "memory_path": str(memory_path),
        "metadata_path": str(metadata_path),
        "source_count": len(source_names),
        "sources": source_names,
        "memory_bank_patches": int(memory_bank.shape[0]),
        "model_name": runtime["model_name"],
        "image_size": runtime["image_size"],
        "patch_grid": [runtime["grid_height"], runtime["grid_width"]],
    }


def build_vision_result(
    payload: dict[str, Any],
    demo_package: Path,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    image_path = payload.get("image_path") or payload.get("vision", {}).get("image_path")
    image_file = Path(image_path).expanduser().resolve() if isinstance(image_path, str) else None
    predicted_fault = infer_fault_from_path(image_path, "visual_anomaly")

    model_mode = "dinov2_patch_memory"
    model_error = ""
    heatmap = heatmap_path_for(run_dir)
    roi_path = roi_path_for(run_dir)
    roi = {"path": str(image_file) if image_file else image_path, "mode": None, "box": None, "error": ""}
    metrics = {
        "anomaly_score": 0.0,
        "is_anomaly": False,
        "threshold": 0.35,
        "image_size": [],
    }
    if image_file is None or not image_file.is_file():
        model_mode = "missing_image"
        model_error = "No readable image file was provided."
    else:
        try:
            roi = crop_asset_region(image_file, roi_path, require_semantic_asset=True)
            if roi.get("mode") == "rejected_no_mechanical_asset":
                model_mode = "rejected_non_mechanical_image"
                model_error = str(roi.get("error", "Image was rejected before DINOv2 scoring."))
                metrics = {
                    "anomaly_score": 0.0,
                    "is_anomaly": False,
                    "threshold": 0.35,
                    "image_size": [],
                }
            else:
                score_path = Path(roi["path"]) if roi.get("path") else image_file
                metrics = score_image(score_path, heatmap, predicted_fault)
        except Exception as exc:
            if model_mode != "rejected_non_mechanical_image":
                model_mode = "image_processing_failed"
                model_error = f"{type(exc).__name__}: {exc}"
                roi = {"path": str(image_file), "mode": "failed", "box": None, "error": model_error}

    return {
        "scenario_id": payload.get("scenario_id", "SCENARIO-001"),
        "asset_id": payload.get("asset_id", "ASSET-001"),
        "anomaly_score": metrics["anomaly_score"],
        "is_anomaly": metrics["is_anomaly"],
        "predicted_fault": predicted_fault,
        "severity": "warning" if metrics["is_anomaly"] else "normal",
        "location": "generated_image_heatmap",
        "image_path": str(image_file) if image_file else image_path,
        "processed_image_path": roi.get("path") if image_file else None,
        "heatmap_path": str(heatmap) if heatmap else None,
        "evidence": {
            "model_mode": model_mode,
            "model_error": model_error,
            "model": metrics.get("model"),
            "model_source": metrics.get("model_source"),
            "image_threshold": metrics["threshold"],
            "pixel_threshold": metrics.get("pixel_threshold"),
            "threshold": metrics["threshold"],
            "image_size": metrics["image_size"],
            "normal_memory_sources": metrics.get("normal_memory_sources", []),
            "evaluation_mode": metrics.get("evaluation_mode"),
            "patch_grid": metrics.get("patch_grid", []),
            "roi_mode": roi.get("mode") if image_file else None,
            "roi_box": roi.get("box") if image_file else None,
            "roi_error": roi.get("error") if image_file else None,
            "roi_detector_label": roi.get("detected_label") if image_file else None,
            "roi_detector_score": roi.get("detector_score") if image_file else None,
            "roi_gpu_backend": roi.get("gpu_backend") if image_file else None,
            "roi_detector_source": roi.get("detector_source") if image_file else None,
            "roi_sam_source": roi.get("sam_source") if image_file else None,
        },
        "limitations": [
            "DINOv2 detects and localizes anomalies but does not classify defect type.",
            "Fault labels are inferred from supplied file/context names unless a classifier is added.",
            "The normal memory bank was built from the synthetic mechanical demo data.",
        ],
    }


def build_rca_result(payload: dict[str, Any], vision: dict[str, Any]) -> dict[str, Any]:
    telemetry = payload.get("telemetry") if isinstance(payload.get("telemetry"), dict) else {}
    rag = payload.get("rag") if isinstance(payload.get("rag"), dict) else {}
    rag_results = rag.get("results") if isinstance(rag.get("results"), list) else []
    citation_ids = [
        str(result.get("document_id"))
        for result in rag_results
        if result.get("document_id")
    ]
    telemetry_risk = float(
        telemetry.get("failure_risk", payload.get("telemetry_risk", 0.0))
    )
    telemetry_rul = float(telemetry.get("predicted_rul", payload.get("predicted_rul", 125.0)))
    vision_anomaly = bool(vision.get("is_anomaly"))
    vision_score = float(vision.get("anomaly_score", 0.0))

    if telemetry_risk >= 0.75 and vision_anomaly:
        status = "critical"
        root_cause = "Telemetry degradation aligns with a strong visual anomaly."
        confidence = min(0.95, 0.7 * telemetry_risk + 0.3 * vision_score)
    elif vision_anomaly:
        status = "warning"
        root_cause = "A visual surface anomaly is present without corroborating critical telemetry."
        confidence = min(0.85, 0.5 * telemetry_risk + 0.5 * vision_score + 0.1)
    elif telemetry_risk >= 0.5:
        status = "warning"
        root_cause = "Telemetry suggests elevated risk before a visual defect is confirmed."
        confidence = min(0.8, telemetry_risk + 0.1)
    else:
        status = "normal"
        root_cause = "No strong telemetry or vision anomaly is present."
        confidence = max(0.2, 1.0 - telemetry_risk)

    if status == "critical":
        action_text = "Stop the asset and inspect the localized visual defect region."
    elif status == "warning":
        action_text = "Inspect the anomaly region and compare it with a clean reference before assigning a fault class."
    else:
        action_text = "Continue monitoring and re-evaluate on the next cycle or image capture."

    if rag_results:
        top_doc = rag_results[0]
        action_text = f"{action_text} Consult {top_doc.get('document_id')} ({top_doc.get('title')})."

    actions = [{
        "priority": 1,
        "action": action_text,
        "citations": citation_ids[:3],
    }]

    return {
        "scenario_id": payload.get("scenario_id", vision.get("scenario_id", "SCENARIO-001")),
        "asset_id": payload.get("asset_id", vision.get("asset_id", "ASSET-001")),
        "status": status,
        "root_cause": root_cause,
        "fault_location": vision.get("location"),
        "confidence": confidence,
        "evidence": [
            {
                "source": "telemetry",
                "observation": f"Predicted RUL is {telemetry_rul:.1f} cycles with failure risk {telemetry_risk:.3f}.",
            },
            {
                "source": "vision",
                "observation": f"Vision branch indicates {vision.get('predicted_fault')} with score {vision_score:.3f}.",
            },
            {
                "source": "rag",
                "observation": (
                    f"Retrieved {len(rag_results)} maintenance passages: "
                    f"{', '.join(citation_ids[:3]) or 'none'}."
                ),
            },
        ],
        "recommended_actions": actions,
        "citations": citation_ids,
        "alternative_hypotheses": [
            "Benign surface variation",
            "Localized wear or damage",
            "Vision-domain false positive",
        ],
        "limitations": vision.get("limitations", []) + [
            "RAG passages are advisory maintenance references, not autonomous instructions.",
        ],
        "safety": {
            "advisory_only": True,
            "autonomous_control": False,
            "qualified_person_required": True,
        },
    }


def predict_vision(
    payload: dict[str, Any],
    demo_package: Path,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    return build_vision_result(payload, demo_package, run_dir)


def fuse_results(telemetry: dict[str, Any], vision: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return build_rca_result({"telemetry": telemetry, **payload}, vision)
