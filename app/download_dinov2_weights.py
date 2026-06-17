from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "facebook/dinov2-base"
DEFAULT_TARGET = APP_ROOT / "backend" / "vision_dinov2" / "facebook_dinov2_base"


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DINOv2 model files into the app backend model folder."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if model.safetensors already exists.",
    )
    args = parser.parse_args()

    target = Path(args.target).expanduser().resolve()
    weight_file = target / "model.safetensors"
    if weight_file.is_file() and not args.force:
        print(f"DINOv2 weights already exist: {weight_file}")
        print("Use --force to download again.")
        return

    ensure_huggingface_hub()
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=target,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
        ],
    )

    print("DINOv2 download complete.")
    print(f"Repo:   {args.repo_id}")
    print(f"Target: {target}")
    print(f"Weights: {weight_file}")


if __name__ == "__main__":
    main()
