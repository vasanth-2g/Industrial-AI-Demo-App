from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
VISION_ROOT = APP_ROOT / "backend" / "vision_dinov2"
DEFAULT_REPO_ID = "facebook/dinov2-base"
DEFAULT_TARGET = VISION_ROOT / "facebook_dinov2_base"
ROI_MODELS = {
    "owlvit": {
        "repo_id": "google/owlvit-base-patch32",
        "target": VISION_ROOT / "google_owlvit_base_patch32",
        "required": ["model.safetensors", "config.json", "preprocessor_config.json"],
    },
    "sam": {
        "repo_id": "facebook/sam-vit-base",
        "target": VISION_ROOT / "facebook_sam_vit_base",
        "required": ["model.safetensors", "config.json", "preprocessor_config.json"],
    },
}


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"],
            check=True,
        )


def has_required_files(target: Path, required: list[str]) -> bool:
    return all((target / name).is_file() for name in required)


def download_snapshot(repo_id: str, target: Path, required: list[str], force: bool) -> None:
    if has_required_files(target, required) and not force:
        print(f"Already exists: {target}")
        return

    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.safetensors",
            "tokenizer.*",
            "vocab.*",
            "merges.txt",
        ],
    )
    print(f"Downloaded {repo_id} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DINOv2 plus optional OWL-ViT/SAM vision model files into app/backend/vision_dinov2."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument(
        "--only-dinov2",
        action="store_true",
        help="Download only facebook/dinov2-base, not OWL-ViT and SAM.",
    )
    parser.add_argument(
        "--skip-dinov2",
        action="store_true",
        help="Download only OWL-ViT and SAM ROI models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if model.safetensors already exists.",
    )
    args = parser.parse_args()

    ensure_huggingface_hub()

    target = Path(args.target).expanduser().resolve()
    if not args.skip_dinov2:
        download_snapshot(
            args.repo_id,
            target,
            ["model.safetensors", "config.json", "preprocessor_config.json"],
            args.force,
        )

    if not args.only_dinov2:
        for model in ROI_MODELS.values():
            download_snapshot(
                str(model["repo_id"]),
                Path(model["target"]),
                list(model["required"]),
                args.force,
            )

    print("Vision model download complete.")
    print(f"DINOv2: {target}")
    if not args.only_dinov2:
        print(f"OWL-ViT: {ROI_MODELS['owlvit']['target']}")
        print(f"SAM:     {ROI_MODELS['sam']['target']}")


if __name__ == "__main__":
    main()
