from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_rca_runtime import generate_llm_rca
from model_config import config_value, resolve_app_path
from rag_runtime import retrieve as retrieve_knowledge
from vision_runtime import predict_vision as predict_vision_result


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TELEMETRY_MODEL_DIR = Path(__file__).resolve().parent / "XGboost" / "telemetry_risk"
TELEMETRY_MODEL_DIR = resolve_app_path(
    config_value("telemetry", "model_dir"),
    DEFAULT_TELEMETRY_MODEL_DIR,
)
RAW_COLUMNS = (
    ["unit", "cycle"]
    + [f"op_setting_{index}" for index in range(1, 4)]
    + [f"sensor_{index}" for index in range(1, 22)]
)
ROLLING_WINDOWS = (5, 10, 20)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_telemetry_dirs(demo_package: Path) -> list[Path]:
    return [TELEMETRY_MODEL_DIR]


def first_file(candidates: list[Path], name: str) -> Path | None:
    for directory in candidates:
        path = directory / name
        if path.is_file():
            return path
    return None


def default_demo_package(repo_root: Path) -> Path:
    return (repo_root / "app").resolve()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def scenario_history(payload: dict[str, Any]) -> list[dict[str, Any]]:
    telemetry = payload.get("telemetry")
    history = payload.get("telemetry_history")

    if isinstance(history, list) and history:
        rows = [row for row in history if isinstance(row, dict)]
        if rows:
            return rows

    if isinstance(telemetry, dict):
        if isinstance(telemetry.get("history"), list) and telemetry["history"]:
            rows = [row for row in telemetry["history"] if isinstance(row, dict)]
            if rows:
                return rows
        if any(key in telemetry for key in RAW_COLUMNS):
            return [telemetry]

    return []


def normalize_history(history: list[dict[str, Any]]) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for index, row in enumerate(history, start=1):
        item: dict[str, float] = {}
        item["unit"] = as_float(row.get("unit", 1))
        item["cycle"] = as_float(row.get("cycle", index))
        for column in RAW_COLUMNS[2:]:
            item[column] = as_float(row.get(column, 0.0))
        normalized.append(item)
    normalized.sort(key=lambda row: (row["unit"], row["cycle"]))
    return normalized


def extract_sensor_names(feature_columns: list[str]) -> list[str]:
    sensors: list[str] = []
    for index in range(1, 22):
        name = f"sensor_{index}"
        if any(column == name or column.startswith(f"{name}_") for column in feature_columns):
            sensors.append(name)
    return sensors


def build_feature_row(
    history: list[dict[str, float]],
    feature_columns: list[str],
    subset: str = "FD001",
) -> dict[str, float]:
    sensors = extract_sensor_names(feature_columns)
    last = history[-1]
    feature_row: dict[str, float] = {}

    feature_row["cycle"] = last["cycle"]
    for index in range(1, 4):
        feature_row[f"op_setting_{index}"] = last.get(f"op_setting_{index}", 0.0)

    for sensor in sensors:
        feature_row[sensor] = last.get(sensor, 0.0)
        previous = history[-2].get(sensor, feature_row[sensor]) if len(history) > 1 else feature_row[sensor]
        feature_row[f"{sensor}_delta"] = feature_row[sensor] - previous

        values = [row.get(sensor, 0.0) for row in history]
        for window in ROLLING_WINDOWS:
            tail = values[-window:]
            mean = sum(tail) / max(len(tail), 1)
            variance = sum((value - mean) ** 2 for value in tail) / max(len(tail), 1)
            feature_row[f"{sensor}_mean_{window}"] = mean
            feature_row[f"{sensor}_std_{window}"] = variance ** 0.5

    for column in feature_columns:
        if column.startswith("subset_"):
            feature_row[column] = 1.0 if column == f"subset_{subset}" else 0.0

    for column in feature_columns:
        feature_row.setdefault(column, 0.0)

    return {column: feature_row.get(column, 0.0) for column in feature_columns}


def severity_from_scores(rul: float, risk: float, telemetry_meta: dict[str, Any]) -> str:
    critical_rul = float(telemetry_meta.get("critical_rul", 15))
    warning_rul = float(telemetry_meta.get("warning_rul", 60))
    alert_threshold = float(telemetry_meta.get("alert_threshold", 0.5))

    if rul <= critical_rul or risk >= max(alert_threshold, 0.85):
        return "critical"
    if rul <= warning_rul or risk >= alert_threshold:
        return "warning"
    return "normal"


def top_feature_attribution(model: Any, feature_frame: Any, feature_columns: list[str], limit: int = 8) -> dict[str, Any]:
    try:
        import xgboost as xgb  # type: ignore

        booster = model.get_booster() if hasattr(model, "get_booster") else model
        matrix = xgb.DMatrix(feature_frame, feature_names=feature_columns)
        contributions = booster.predict(matrix, pred_contribs=True)[0].tolist()
    except Exception as exc:
        return {
            "mode": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "top_features": [],
            "top_sensors": [],
        }

    feature_contribs = []
    sensor_scores: dict[str, float] = {}
    for name, contribution in zip(feature_columns, contributions[: len(feature_columns)]):
        value = float(feature_frame.iloc[0][name])
        abs_contribution = abs(float(contribution))
        feature_contribs.append(
            {
                "feature": name,
                "value": value,
                "contribution": float(contribution),
                "abs_contribution": abs_contribution,
            }
        )
        for index in range(1, 22):
            sensor = f"sensor_{index}"
            if name == sensor or name.startswith(f"{sensor}_"):
                sensor_scores[sensor] = sensor_scores.get(sensor, 0.0) + abs_contribution
                break

    top_features = sorted(feature_contribs, key=lambda item: item["abs_contribution"], reverse=True)[:limit]
    top_sensors = [
        {"sensor": sensor, "importance": score}
        for sensor, score in sorted(sensor_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]
    return {
        "mode": "xgboost_pred_contribs",
        "error": "",
        "top_features": top_features,
        "top_sensors": top_sensors,
    }


def predict_telemetry(
    payload: dict[str, Any],
    demo_package: Path,
) -> dict[str, Any]:
    telemetry_dirs = candidate_telemetry_dirs(demo_package)
    metadata = load_json(TELEMETRY_MODEL_DIR / str(config_value("telemetry", "metadata_file", "metadata.json")))

    history = normalize_history(scenario_history(payload))
    feature_columns = metadata.get("feature_columns", [])

    model_mode = "unavailable"
    model_error = ""
    rul = 125.0
    risk = 0.0
    threshold = float(metadata.get("alert_threshold", 0.5))
    attribution = {
        "mode": "unavailable",
        "error": "",
        "top_features": [],
        "top_sensors": [],
    }

    try:
        import joblib  # type: ignore

        rul_model_path = first_file(telemetry_dirs, str(config_value("telemetry", "rul_model_file", "xgb_rul_model.joblib")))
        risk_model_path = first_file(telemetry_dirs, str(config_value("telemetry", "risk_model_file", "xgb_risk_model.joblib")))
        feature_file = first_file(telemetry_dirs, str(config_value("telemetry", "feature_columns_file", "feature_columns.json")))
        if feature_file is not None:
            feature_columns = json.loads(feature_file.read_text(encoding="utf-8"))

        if history and rul_model_path is not None and risk_model_path is not None and feature_columns:
            row = build_feature_row(history, feature_columns, subset=str(payload.get("subset", "FD001")))
            feature_frame = __import__("pandas").DataFrame([row], columns=feature_columns)  # type: ignore
            rul_model = joblib.load(rul_model_path)
            risk_model = joblib.load(risk_model_path)
            rul = float(rul_model.predict(feature_frame)[0])
            if hasattr(risk_model, "predict_proba"):
                risk = float(risk_model.predict_proba(feature_frame)[:, 1][0])
            else:
                risk = float(risk_model.predict(feature_frame)[0])
            attribution = top_feature_attribution(risk_model, feature_frame, feature_columns)
            model_mode = "runtime_xgboost"
        elif history:
            model_error = "Telemetry history was provided, but model files or feature columns were missing."
        else:
            model_error = "No telemetry history was provided."
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"

    severity = severity_from_scores(rul, risk, metadata)
    risk_label = int(risk >= threshold)
    last_cycle = history[-1]["cycle"] if history else None

    telemetry = {
        "scenario_id": payload.get("scenario_id", "SCENARIO-001"),
        "asset_id": payload.get("asset_id", "ASSET-001"),
        "anomaly_score": risk,
        "failure_risk": risk,
        "predicted_rul": rul,
        "severity": severity,
        "alert": bool(severity != "normal"),
        "risk_label": risk_label,
        "evidence": {
            "model_mode": model_mode,
            "model_error": model_error,
            "alert_threshold": threshold,
            "feature_count": len(feature_columns),
            "history_length": len(history),
            "last_observed_cycle": last_cycle,
            "feature_attribution": attribution,
            "top_sensors": attribution.get("top_sensors", []),
            "artifact_dirs_checked": [str(path) for path in telemetry_dirs],
        },
        "limitations": [
            "Telemetry prediction uses the trained XGBoost artifacts in app/backend/XGboost/telemetry_risk.",
            "Input sensor rows must match the C-MAPSS FD001-style schema used during training.",
        ],
    }
    return telemetry


def fuse_results(telemetry: dict[str, Any], vision: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    telemetry_risk = float(telemetry.get("failure_risk", 0.0))
    vision_score = float(vision.get("anomaly_score", 0.0))
    telemetry_severity = telemetry.get("severity", "normal")
    vision_anomaly = bool(vision.get("is_anomaly"))

    if telemetry_severity == "critical" or (telemetry_risk >= 0.75 and vision_anomaly):
        status = "critical"
        confidence = min(0.95, 0.75 * telemetry_risk + 0.25 * vision_score + 0.2)
        root_cause = "Telemetry and vision both indicate a high-probability failure condition."
    elif telemetry_severity == "warning" or vision_anomaly:
        status = "warning"
        confidence = min(0.85, 0.5 * telemetry_risk + 0.5 * vision_score + 0.1)
        if vision_anomaly and telemetry_risk >= 0.5:
            root_cause = "Telemetry degradation aligns with a localized visual anomaly."
        elif vision_anomaly:
            root_cause = "A visual surface anomaly is present without corroborating critical telemetry."
        else:
            root_cause = "Telemetry suggests elevated risk before a visual defect is confirmed."
    else:
        status = "normal"
        confidence = max(0.2, 1.0 - telemetry_risk)
        root_cause = "No strong telemetry or vision anomaly is present."

    actions = []
    if status == "critical":
        actions.append(
            {
                "priority": 1,
                "action": "Stop the asset and inspect both telemetry and the localized visual defect region.",
            }
        )
    elif status == "warning":
        actions.append(
            {
                "priority": 1,
                "action": "Inspect the anomaly region and compare it with a clean reference before assigning a fault class.",
            }
        )
    else:
        actions.append(
            {
                "priority": 1,
                "action": "Continue monitoring and re-evaluate on the next cycle or image capture.",
            }
        )

    return {
        "scenario_id": payload.get("scenario_id", telemetry.get("scenario_id", "SCENARIO-001")),
        "asset_id": payload.get("asset_id", telemetry.get("asset_id", "ASSET-001")),
        "timestamp": payload.get("timestamp"),
        "status": status,
        "root_cause": root_cause,
        "fault_location": vision.get("location"),
        "confidence": confidence,
        "evidence": [
            {
                "source": "telemetry",
                "observation": f"Predicted RUL is {telemetry.get('predicted_rul'):.1f} cycles with failure risk {telemetry_risk:.3f}.",
            },
            {
                "source": "vision",
                "observation": f"DINOv2-style visual evidence indicates {vision.get('predicted_fault')}.",
            },
        ],
        "alternative_hypotheses": [
            "Benign surface variation",
            "Localized wear or damage",
            "Vision-domain false positive",
        ],
        "recommended_actions": actions,
        "citations": payload.get("citations", ["SOP-SAFE-001", "GUIDE-CRK-003"]),
        "limitations": telemetry.get("limitations", []) + vision.get("limitations", []),
        "safety": {
            "advisory_only": True,
            "autonomous_control": False,
            "qualified_person_required": True,
        },
    }


def run_scenario(payload: dict[str, Any], demo_package: Path, run_dir: Path | None = None) -> dict[str, Any]:
    vision = predict_vision_result(payload, demo_package, run_dir)
    if vision.get("evidence", {}).get("model_mode") == "rejected_non_mechanical_image":
        return {
            "error": "image_rejected",
            "message": vision.get("evidence", {}).get(
                "model_error",
                "No robot or mechanical asset was detected in the uploaded image.",
            ),
            "vision": vision,
            "telemetry": None,
            "rag": None,
            "llm_rca": None,
            "fusion": None,
        }

    telemetry = predict_telemetry(payload, demo_package)
    rag = retrieve_knowledge(payload, telemetry, vision)
    fusion = generate_llm_rca(payload, telemetry, vision, rag)
    return {
        "telemetry": telemetry,
        "vision": vision,
        "rag": rag,
        "llm_rca": fusion,
        "fusion": fusion,
    }
