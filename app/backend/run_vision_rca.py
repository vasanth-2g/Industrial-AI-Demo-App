from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_runtime import default_demo_package, fuse_results, predict_vision


def load_payload(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the vision branch and fuse it into RCA output."
    )
    parser.add_argument(
        "--payload",
        type=Path,
        help="Path to a scenario JSON file.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        help="Relative or absolute image path to include in the scenario.",
    )
    parser.add_argument(
        "--asset-id",
        type=str,
        default="ASSET-001",
        help="Asset identifier for the scenario.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="SCENARIO-001",
        help="Scenario identifier for the scenario.",
    )
    parser.add_argument(
        "--demo-package",
        type=Path,
        help="Override the demo package path.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    demo_package = args.demo_package or default_demo_package(repo_root)
    payload = load_payload(args.payload)

    if args.image_path:
        payload["image_path"] = args.image_path
    payload.setdefault("scenario_id", args.scenario_id)
    payload.setdefault("asset_id", args.asset_id)

    vision = predict_vision(payload, demo_package)
    fusion = fuse_results(
        {"scenario_id": payload["scenario_id"], "asset_id": payload["asset_id"]},
        vision,
        payload,
    )

    print(json.dumps({"vision": vision, "rca": fusion}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
