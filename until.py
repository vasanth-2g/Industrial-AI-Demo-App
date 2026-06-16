import json
from pathlib import Path


OUTPUT = Path("note6_multimodal_fusion_rule_rca.ipynb")


def lines(text):
    return [line + "\n" for line in text.strip("\n").splitlines()]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    md(
        """
# Multimodal Evidence Fusion and Rule-Based RCA

This notebook completes the first end-to-end intelligence baseline:

```text
C-MAPSS telemetry contract
        +
DINOv2 vision contract
        +
maintenance RAG contract
        ↓
normalized evidence
        ↓
deterministic RCA and safe recommendations
        ↓
structured JSON for the dashboard
```

No LLM and no fine-tuning are required. The output is traceable and provides a
fallback for any later reasoning model.

Important limitation: C-MAPSS turbofan telemetry and synthetic robot-joint
images are separate datasets joined into one scripted demonstration scenario.
The notebook does not claim they are observations of the same physical asset.
"""
    ),
    md("## 1. Paths and Configuration"),
    code(
        """
from pathlib import Path
from datetime import datetime, timezone
import json
import math

import pandas as pd

SCENARIO_ID = "DEMO-MULTIMODAL-001"
DEMO_ASSET_ID = "ROBOT-JOINT-DEMO-01"

TELEMETRY_PATH = Path(
    "/workspace/notebooks/telemetrics/artifacts/cmapss/fd001/"
    "risk/telemetry_contract_example.json"
)
VISION_PATH = Path(
    "/workspace/notebooks/vision/artifacts/synthetic_mechanical/"
    "dinov2/vision_contract_example.json"
)
RAG_PATH = Path(
    "/workspace/notebooks/knowledge/artifacts/rag/"
    "rag_contract_example.json"
)
OUTPUT_ROOT = Path(
    "/workspace/notebooks/integration/artifacts/"
    "multimodal_rule_rca"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Telemetry:", TELEMETRY_PATH)
print("Vision:", VISION_PATH)
print("RAG:", RAG_PATH)
print("Output:", OUTPUT_ROOT)
"""
    ),
    md("## 2. Load and Validate Component Contracts"),
    code(
        """
def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


telemetry = load_json(TELEMETRY_PATH)
vision = load_json(VISION_PATH)
rag = load_json(RAG_PATH)

telemetry_required = {
    "scenario_id", "asset_id", "failure_risk", "predicted_rul",
    "severity", "evidence", "limitations",
}
vision_required = {
    "scenario_id", "asset_id", "anomaly_score", "is_anomaly",
    "predicted_fault", "location", "evidence", "limitations",
}
rag_required = {
    "query_id", "scenario_id", "query", "results", "limitations",
}

for name, payload, required in [
    ("telemetry", telemetry, telemetry_required),
    ("vision", vision, vision_required),
    ("rag", rag, rag_required),
]:
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{name} contract missing: {sorted(missing)}")

known_citation_ids = {
    result["document_id"] for result in rag["results"]
}
if not known_citation_ids:
    raise ValueError("RAG contract contains no retrieved documents.")

print("Contracts loaded successfully.")
print("Telemetry source scenario:", telemetry["scenario_id"])
print("Vision source scenario:", vision["scenario_id"])
print("Retrieved citations:", sorted(known_citation_ids))
"""
    ),
    md("## 3. Normalize Evidence"),
    code(
        """
def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, float(value)))


failure_risk = clamp(telemetry["failure_risk"])
predicted_rul = max(0.0, float(telemetry["predicted_rul"]))
vision_detected = bool(vision["is_anomaly"])
vision_score = max(0.0, float(vision["anomaly_score"]))

# DINOv2 score is a distance, not a calibrated probability.
vision_evidence_strength = 0.65 if vision_detected else 0.25
telemetry_evidence_strength = failure_risk

modalities_present = {
    "telemetry": telemetry is not None,
    "vision": vision is not None,
    "retrieval": bool(rag["results"]),
}

limitations = [
    *telemetry.get("limitations", []),
    *vision.get("limitations", []),
    *rag.get("limitations", []),
    (
        "Telemetry and vision originate from separate demonstration datasets; "
        "their fusion validates system integration, not physical correlation."
    ),
    (
        "The visual anomaly score is not a calibrated probability and does "
        "not identify a specific crack, corrosion, leak, or wear class."
    ),
]
limitations = list(dict.fromkeys(limitations))

normalized_evidence = {
    "scenario_id": SCENARIO_ID,
    "asset_id": DEMO_ASSET_ID,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "synthetic_demo": True,
    "telemetry": {
        "source_scenario_id": telemetry["scenario_id"],
        "source_asset_id": telemetry["asset_id"],
        "failure_risk": failure_risk,
        "predicted_rul": predicted_rul,
        "severity": telemetry["severity"],
        "alert": bool(telemetry.get("alert", failure_risk >= 0.5)),
        "evidence": telemetry["evidence"],
    },
    "vision": {
        "source_scenario_id": vision["scenario_id"],
        "source_asset_id": vision["asset_id"],
        "anomaly_detected": vision_detected,
        "anomaly_score_raw": vision_score,
        "score_type": "uncalibrated_dinov2_patch_distance",
        "predicted_fault": vision["predicted_fault"],
        "location": vision["location"],
        "image_path": vision.get("image_path"),
    },
    "retrieval": {
        "query_id": rag["query_id"],
        "query": rag["query"],
        "results": rag["results"],
    },
    "evidence_quality": {
        "modalities_present": modalities_present,
        "telemetry_strength": telemetry_evidence_strength,
        "vision_strength": vision_evidence_strength,
        "retrieval_count": len(rag["results"]),
        "independent_physical_pair": False,
    },
    "limitations": limitations,
}

display(pd.DataFrame([
    {
        "failure_risk": failure_risk,
        "predicted_rul": predicted_rul,
        "telemetry_severity": telemetry["severity"],
        "visual_anomaly": vision_detected,
        "visual_score_raw": vision_score,
        "retrieved_documents": len(rag["results"]),
    }
]))
"""
    ),
    md("## 4. Deterministic Evidence Fusion"),
    code(
        """
def status_from_evidence(risk, rul, visual_anomaly):
    if risk >= 0.80 or rul <= 15:
        return "critical"
    if risk >= 0.50 or rul <= 60 or visual_anomaly:
        return "warning"
    return "normal"


status = status_from_evidence(
    failure_risk, predicted_rul, vision_detected
)

agreement = (
    bool(telemetry.get("alert", failure_risk >= 0.5))
    and vision_detected
)

# Confidence describes support for "inspection required", not a specific fault.
confidence = (
    0.50 * telemetry_evidence_strength
    + 0.25 * vision_evidence_strength
    + 0.15 * min(len(rag["results"]) / 3.0, 1.0)
    + 0.10 * float(agreement)
)
confidence = clamp(confidence)

# Cap confidence because the modalities are demo-linked independent datasets.
confidence = min(confidence, 0.78)

fusion = {
    "status": status,
    "modalities_agree_on_abnormal_condition": agreement,
    "confidence": confidence,
    "requires_inspection": status in {"warning", "critical"},
    "autonomous_action_allowed": False,
}

display(pd.DataFrame([fusion]))
"""
    ),
    md("## 5. Rule-Based Root-Cause Analysis"),
    code(
        """
retrieved_by_id = {
    result["document_id"]: result
    for result in rag["results"]
}


def available_citations(preferred_ids):
    return [
        document_id
        for document_id in preferred_ids
        if document_id in retrieved_by_id
    ]


telemetry_alert = bool(telemetry.get("alert", failure_risk >= 0.5))

if status == "critical" and telemetry_alert and vision_detected:
    root_cause = (
        "Mechanical degradation is suspected, with an unresolved visual "
        "surface anomaly requiring inspection."
    )
    alternative_hypotheses = [
        "Bearing or gearbox degradation",
        "Misalignment or loose mounting",
        "Sensor/model mismatch between independent demo modalities",
    ]
elif status in {"critical", "warning"} and telemetry_alert:
    root_cause = (
        "Telemetry indicates degradation risk; the physical cause remains "
        "unconfirmed."
    )
    alternative_hypotheses = [
        "Bearing or gearbox degradation",
        "Abnormal load",
        "Sensor drift or model mismatch",
    ]
elif vision_detected:
    root_cause = (
        "A visual surface anomaly is present without corroborating critical "
        "telemetry."
    )
    alternative_hypotheses = [
        "Benign surface variation",
        "Localized wear or damage",
        "Vision-domain false positive",
    ]
else:
    root_cause = "No critical degradation is supported by current evidence."
    alternative_hypotheses = []

evidence_items = [
    {
        "source": "telemetry",
        "observation": (
            f"Predicted RUL is {predicted_rul:.1f} cycles with "
            f"failure risk {failure_risk:.3f}."
        ),
    },
    {
        "source": "vision",
        "observation": (
            "DINOv2 detected a visual anomaly and produced patch-level "
            "localization."
            if vision_detected
            else "DINOv2 did not detect a visual anomaly."
        ),
    },
    {
        "source": "retrieval",
        "observation": (
            f"{len(rag['results'])} maintenance passages were retrieved."
        ),
    },
]

citations = available_citations([
    "SOP-SAFE-001",
    "SOP-GBX-004",
    "ERR-E204",
    "CHECK-ALIGN-009",
    "GUIDE-CRK-003",
])

if status == "critical":
    recommended_actions = [
        {
            "priority": 1,
            "action": (
                "Place the demonstration asset in a safe state and prevent "
                "automatic restart pending qualified inspection."
            ),
            "citations": available_citations(["SOP-SAFE-001"]),
        },
        {
            "priority": 2,
            "action": (
                "Inspect gearbox/bearing condition, lubrication, alignment, "
                "mounting, and abnormal vibration sources."
            ),
            "citations": available_citations([
                "SOP-GBX-004", "ERR-E204", "CHECK-ALIGN-009"
            ]),
        },
        {
            "priority": 3,
            "action": (
                "Inspect the localized surface region and use an appropriate "
                "confirmation method before assigning a defect class."
            ),
            "citations": available_citations(["GUIDE-CRK-003"]),
        },
    ]
elif status == "warning" and telemetry_alert:
    recommended_actions = [
        {
            "priority": 1,
            "action": "Schedule inspection and increase monitoring frequency.",
            "citations": available_citations([
                "SOP-GBX-004", "ERR-E204"
            ]),
        }
    ]
elif status == "warning" and vision_detected:
    recommended_actions = [
        {
            "priority": 1,
            "action": (
                "Inspect the localized visual anomaly region and compare it "
                "with a clean reference image before assigning a defect class."
            ),
            "citations": available_citations([
                "SOP-SAFE-001", "GUIDE-CRK-003"
            ]),
        }
    ]
else:
    recommended_actions = [
        {
            "priority": 1,
            "action": "Continue monitoring under the approved operating plan.",
            "citations": [],
        }
    ]

rca_output = {
    "scenario_id": SCENARIO_ID,
    "asset_id": DEMO_ASSET_ID,
    "timestamp": normalized_evidence["timestamp"],
    "synthetic_demo": True,
    "status": status,
    "root_cause": root_cause,
    "fault_location": (
        "localized_patch_region" if vision_detected else "unknown"
    ),
    "confidence": confidence,
    "evidence": evidence_items,
    "alternative_hypotheses": alternative_hypotheses,
    "recommended_actions": recommended_actions,
    "citations": citations,
    "limitations": limitations,
    "safety": {
        "advisory_only": True,
        "autonomous_control": False,
        "qualified_person_required": True,
    },
}

print(json.dumps(rca_output, indent=2))
"""
    ),
    md("## 6. Validate the RCA Contract and Citations"),
    code(
        """
required_rca_fields = {
    "scenario_id", "asset_id", "status", "root_cause",
    "fault_location", "confidence", "evidence",
    "alternative_hypotheses", "recommended_actions",
    "citations", "limitations", "safety",
}
missing = required_rca_fields - set(rca_output)
if missing:
    raise ValueError(f"RCA output missing fields: {sorted(missing)}")

if rca_output["status"] not in {"normal", "warning", "critical"}:
    raise ValueError("Invalid RCA status.")
if not 0.0 <= rca_output["confidence"] <= 1.0:
    raise ValueError("RCA confidence must be between zero and one.")
if not set(rca_output["citations"]).issubset(known_citation_ids):
    raise ValueError("RCA contains a citation not returned by RAG.")

for action in rca_output["recommended_actions"]:
    if not set(action["citations"]).issubset(known_citation_ids):
        raise ValueError(
            f"Action {action['priority']} contains an invalid citation."
        )

if rca_output["safety"]["autonomous_control"]:
    raise ValueError("Baseline must not authorize autonomous control.")

print("RCA schema validation: PASSED")
print("Citation validation: PASSED")
print("Safety validation: PASSED")
"""
    ),
    md("## 7. Create the Reasoning-Model Input Package"),
    code(
        """
reasoning_input = {
    "instruction": (
        "Using only the supplied evidence and retrieved passages, produce "
        "structured root-cause hypotheses and safe maintenance actions. "
        "Do not invent a visual defect class or unsupported component failure."
    ),
    "output_schema": {
        "status": "normal|warning|critical",
        "root_cause": "string",
        "fault_location": "string",
        "confidence": "number from 0 to 1",
        "evidence": "array",
        "alternative_hypotheses": "array",
        "recommended_actions": "array",
        "citations": "retrieved document IDs only",
        "limitations": "array",
    },
    "normalized_evidence": normalized_evidence,
    "deterministic_fallback": rca_output,
}

print(
    "Reasoning-model package prepared. "
    "The deterministic RCA remains the fallback."
)
"""
    ),
    md("## 8. Save the End-to-End Baseline Artifacts"),
    code(
        """
paths = {
    "normalized_evidence": OUTPUT_ROOT / "normalized_evidence.json",
    "rule_rca": OUTPUT_ROOT / "rule_rca_output.json",
    "reasoning_input": OUTPUT_ROOT / "reasoning_input.json",
    "trace": OUTPUT_ROOT / "execution_trace.json",
}

paths["normalized_evidence"].write_text(
    json.dumps(normalized_evidence, indent=2),
    encoding="utf-8",
)
paths["rule_rca"].write_text(
    json.dumps(rca_output, indent=2),
    encoding="utf-8",
)
paths["reasoning_input"].write_text(
    json.dumps(reasoning_input, indent=2),
    encoding="utf-8",
)

execution_trace = {
    "scenario_id": SCENARIO_ID,
    "steps": [
        "load_component_contracts",
        "normalize_evidence",
        "fuse_evidence",
        "retrieve_knowledge",
        "generate_rule_rca",
        "validate_schema_citations_and_safety",
        "prepare_reasoning_model_input",
    ],
    "inputs": {
        "telemetry_contract": str(TELEMETRY_PATH),
        "vision_contract": str(VISION_PATH),
        "rag_contract": str(RAG_PATH),
    },
    "outputs": {
        key: str(path) for key, path in paths.items()
        if key != "trace"
    },
    "result": "passed",
}
paths["trace"].write_text(
    json.dumps(execution_trace, indent=2),
    encoding="utf-8",
)

for name, path in paths.items():
    print(f"{name}: {path}")
"""
    ),
    md(
        """
## Completion Criteria

- Telemetry, vision, and RAG contracts load successfully.
- Raw model scores remain distinguishable from calibrated probabilities.
- Separate source datasets are clearly disclosed.
- RCA confidence is capped for demo-linked modalities.
- Every recommendation is advisory and citation constrained.
- RCA, citations, and safety fields pass validation.
- A reasoning-model input package and deterministic fallback are saved.

The next stage can call a pretrained reasoning model without fine-tuning, then
compare its structured answer against this deterministic fallback.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT} with {len(cells)} cells.")
