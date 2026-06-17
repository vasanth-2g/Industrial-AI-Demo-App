from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from model_config import config_value, resolve_app_path

_CONFIG_HF_MODEL = str(config_value("llm", "hf_model", "Qwen/Qwen2.5-0.5B-Instruct"))
_CONFIG_ALLOW_DOWNLOAD = bool(config_value("llm", "allow_model_download", False))
_CONFIG_MAX_NEW_TOKENS = int(config_value("llm", "max_new_tokens", 700))


def resolve_hf_model(value: str) -> str:
    if value.startswith(("backend/", "backend\\", "./", ".\\", "hf_models/", "hf_models\\")) or os.path.isabs(value):
        return str(resolve_app_path(value, Path(".")))
    return value


DEFAULT_HF_MODEL = os.environ.get("HF_LLM_MODEL", resolve_hf_model(_CONFIG_HF_MODEL))
HF_MAX_NEW_TOKENS = int(os.environ.get("HF_LLM_MAX_NEW_TOKENS", str(_CONFIG_MAX_NEW_TOKENS)))
_HF_PIPELINE: Any | None = None
STATUS_RANK = {"normal": 0, "warning": 1, "critical": 2}


def visual_evidence_floor(vision: dict[str, Any]) -> str:
    if not bool(vision.get("is_anomaly")):
        return "normal"

    fault = str(vision.get("predicted_fault", "")).lower()
    score = float(vision.get("anomaly_score", 0.0))
    threshold = float(vision.get("evidence", {}).get("threshold", 0.35))

    if fault == "crack" and score >= max(0.75, threshold):
        return "critical"
    return "warning"


def higher_status(left: str, right: str) -> str:
    left = left if left in STATUS_RANK else "normal"
    right = right if right in STATUS_RANK else "normal"
    return left if STATUS_RANK[left] >= STATUS_RANK[right] else right


def classify(telemetry: dict[str, Any], vision: dict[str, Any]) -> str:
    telemetry_risk = float(telemetry.get("failure_risk", 0.0))
    telemetry_severity = str(telemetry.get("severity", "normal")).lower()
    vision_anomaly = bool(vision.get("is_anomaly"))
    vision_score = float(vision.get("anomaly_score", 0.0))

    floor = visual_evidence_floor(vision)

    if telemetry_severity == "critical" or (telemetry_risk >= 0.75 and vision_anomaly):
        return "critical"
    if telemetry_severity == "warning" or telemetry_risk >= 0.5 or vision_anomaly or vision_score >= 0.5:
        return higher_status("warning", floor)
    return "normal"


def confidence_score(telemetry: dict[str, Any], vision: dict[str, Any], rag: dict[str, Any]) -> float:
    telemetry_risk = float(telemetry.get("failure_risk", 0.0))
    telemetry_severity = str(telemetry.get("severity", "normal")).lower()
    vision_score = float(vision.get("anomaly_score", 0.0))
    vision_anomaly = bool(vision.get("is_anomaly"))
    rag_results = rag.get("results") if isinstance(rag.get("results"), list) else []
    rag_score = float(rag_results[0].get("score", 0.0)) if rag_results else 0.0
    score = 0.30
    score += 0.42 * min(vision_score, 1.0) if vision_anomaly else 0.10 * min(vision_score, 1.0)
    score += 0.18 * min(telemetry_risk, 1.0)
    score += 0.10 * min(rag_score, 1.0)
    if telemetry_severity in {"warning", "critical"} and vision_anomaly:
        score += 0.08
    elif vision_anomaly and telemetry_severity == "normal":
        score -= 0.03
    return round(min(max(score, 0.0), 0.95), 4)


def fallback_rca(
    payload: dict[str, Any],
    telemetry: dict[str, Any],
    vision: dict[str, Any],
    rag: dict[str, Any],
) -> dict[str, Any]:
    rag_results = rag.get("results") if isinstance(rag.get("results"), list) else []
    citations = [result.get("document_id") for result in rag_results if result.get("document_id")]
    status = classify(telemetry, vision)
    fault = vision.get("predicted_fault", "visual_anomaly")
    telemetry_mode = telemetry.get("evidence", {}).get("model_mode", "unknown")

    if status == "critical":
        root_cause = f"Telemetry risk and DINOv2 visual evidence indicate a high-risk {fault} condition."
    elif bool(vision.get("is_anomaly")):
        root_cause = f"DINOv2 detected a localized visual anomaly consistent with {fault}; telemetry is {telemetry.get('severity', 'unknown')}."
    elif telemetry.get("severity") in {"warning", "critical"}:
        root_cause = "Telemetry indicates elevated degradation risk without strong visual corroboration."
    else:
        root_cause = "No high-confidence fault condition is indicated by the current telemetry and vision evidence."

    next_steps = []
    if status == "critical":
        next_steps.append("Stop or isolate the asset according to site safety procedure before inspection.")
    if bool(vision.get("is_anomaly")):
        next_steps.append("Inspect the DINOv2 localized region and compare it with a clean reference image.")
    if telemetry_mode == "runtime_xgboost":
        next_steps.append("Review the latest sensor history and repeat telemetry scoring after the next operating cycle.")
    else:
        next_steps.append("Provide raw telemetry sensor history to corroborate the visual finding with XGBoost scoring.")
    if rag_results:
        top = rag_results[0]
        next_steps.append(f"Follow the retrieved guidance in {top.get('document_id')} ({top.get('title')}).")
    next_steps.append("Record findings, corrective action, and follow-up inspection date.")

    return {
        "scenario_id": payload.get("scenario_id", telemetry.get("scenario_id", vision.get("scenario_id", "SCENARIO-001"))),
        "asset_id": payload.get("asset_id", telemetry.get("asset_id", vision.get("asset_id", "ASSET-001"))),
        "classification": status,
        "status": status,
        "confidence_score": confidence_score(telemetry, vision, rag),
        "confidence": confidence_score(telemetry, vision, rag),
        "root_cause": root_cause,
        "fault_location": vision.get("location"),
        "evidence_summary": {
            "telemetry": {
                "predicted_rul": telemetry.get("predicted_rul"),
                "failure_risk": telemetry.get("failure_risk"),
                "severity": telemetry.get("severity"),
                "model_mode": telemetry.get("evidence", {}).get("model_mode"),
                "top_sensors": telemetry.get("evidence", {}).get("top_sensors", []),
            },
            "vision": {
                "predicted_fault": fault,
                "anomaly_score": vision.get("anomaly_score"),
                "is_anomaly": vision.get("is_anomaly"),
                "model_mode": vision.get("evidence", {}).get("model_mode"),
            },
            "rag": {
                "mode": rag.get("mode"),
                "top_citations": citations[:5],
            },
        },
        "evidence": [
            {
                "source": "telemetry",
                "observation": f"Predicted RUL={telemetry.get('predicted_rul')}; failure risk={telemetry.get('failure_risk')}; severity={telemetry.get('severity')}.",
            },
            {
                "source": "vision",
                "observation": f"DINOv2 anomaly={vision.get('is_anomaly')} with score={vision.get('anomaly_score')} and fault={fault}.",
            },
            {
                "source": "rag",
                "observation": f"Hybrid retrieval returned {len(rag_results)} passages: {', '.join(citations[:3]) or 'none'}.",
            },
        ],
        "next_steps": next_steps,
        "recommended_actions": [
            {
                "priority": index + 1,
                "action": step,
                "citations": citations[:3],
            }
            for index, step in enumerate(next_steps[:5])
        ],
        "citations": citations,
        "alternative_hypotheses": [
            "Benign visual variation or lighting artifact",
            "Localized surface degradation",
            "Sensor drift or missing telemetry history",
        ],
        "limitations": [
            "LLM RCA must only use telemetry, vision, and retrieved RAG evidence.",
            "RAG passages are advisory references and do not authorize autonomous control.",
        ],
        "safety": {
            "advisory_only": True,
            "autonomous_control": False,
            "qualified_person_required": True,
        },
        "llm": {
            "mode": "deterministic_fallback",
            "model": None,
            "error": "",
        },
    }


def llm_prompt(payload: dict[str, Any], telemetry: dict[str, Any], vision: dict[str, Any], rag: dict[str, Any]) -> list[dict[str, str]]:
    compact_rag = [
        {
            "document_id": item.get("document_id"),
            "title": item.get("title"),
            "text": item.get("text"),
            "score": item.get("score"),
        }
        for item in (rag.get("results") or [])[:5]
    ]
    user_content = {
        "scenario": payload,
        "telemetry_output": telemetry,
        "vision_output": vision,
        "hybrid_rag_results": compact_rag,
        "required_json_schema": {
            "classification": "normal|warning|critical",
            "confidence_score": "number 0..1",
            "root_cause": "string",
            "evidence_summary": "object with telemetry, vision, rag",
            "next_steps": "array of strings",
            "recommended_actions": "array of objects with priority, action, citations",
            "citations": "array of retrieved document_id strings only",
            "limitations": "array of strings",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an industrial maintenance RCA assistant. Use only the supplied telemetry, "
                "vision, and retrieved RAG passages. Return strict JSON only. Do not invent citations. "
                "Do not authorize autonomous control."
            ),
        },
        {"role": "user", "content": json.dumps(user_content, indent=2)},
    ]


def load_huggingface_pipeline() -> Any:
    global _HF_PIPELINE
    if _HF_PIPELINE is not None:
        return _HF_PIPELINE

    allow_download = os.environ.get("HF_ALLOW_MODEL_DOWNLOAD", "1" if _CONFIG_ALLOW_DOWNLOAD else "0") == "1"
    if not allow_download and not os.path.isdir(DEFAULT_HF_MODEL):
        try:
            from huggingface_hub import try_to_load_from_cache  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Hugging Face cache check is unavailable: {type(exc).__name__}: {exc}") from exc
        if try_to_load_from_cache(DEFAULT_HF_MODEL, "config.json") is None:
            raise RuntimeError(
                f"Hugging Face model {DEFAULT_HF_MODEL} is not cached locally. "
                "Download it first or set HF_LLM_MODEL to a local model directory."
            )

    try:
        import torch  # type: ignore
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Hugging Face runtime is unavailable: {type(exc).__name__}: {exc}") from exc

    kwargs: dict[str, Any] = {
        "model": DEFAULT_HF_MODEL,
        "trust_remote_code": True,
        "local_files_only": not allow_download,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["device"] = -1

    _HF_PIPELINE = pipeline("text-generation", **kwargs)
    return _HF_PIPELINE


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Hugging Face model did not return a JSON object.")


def call_huggingface(messages: list[dict[str, str]]) -> dict[str, Any]:
    generator = load_huggingface_pipeline()
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)
        prompt += "\n\nASSISTANT:\n"

    output = generator(
        prompt,
        max_new_tokens=HF_MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
        pad_token_id=getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None,
    )
    text = output[0]["generated_text"] if isinstance(output, list) else str(output)
    return extract_json_object(text)


def generate_llm_rca(
    payload: dict[str, Any],
    telemetry: dict[str, Any],
    vision: dict[str, Any],
    rag: dict[str, Any],
) -> dict[str, Any]:
    fallback = fallback_rca(payload, telemetry, vision, rag)
    messages = llm_prompt(payload, telemetry, vision, rag)

    try:
        llm_result = call_huggingface(messages)
        merged = {**fallback, **llm_result}
        llm_status = str(merged.get("classification", fallback["classification"])).lower()
        evidence_status = fallback["classification"]
        final_status = higher_status(llm_status, evidence_status)
        merged["classification"] = final_status
        merged["status"] = final_status
        merged["confidence"] = merged.get("confidence_score", fallback["confidence_score"])
        merged["llm"] = {
            "mode": "huggingface_transformers",
            "model": DEFAULT_HF_MODEL,
            "error": "",
        }
        return merged
    except Exception as exc:
        fallback["llm"] = {
            "mode": "deterministic_fallback",
            "model": DEFAULT_HF_MODEL,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return fallback


def fallback_chat_answer(question: str, context: dict[str, Any], rag: dict[str, Any]) -> dict[str, Any]:
    results = rag.get("results") if isinstance(rag.get("results"), list) else []
    top_score = float(results[0].get("score", 0.0)) if results else 0.0
    telemetry = context.get("telemetry") or {}
    vision = context.get("vision") or {}
    question_lower = question.lower()
    fault = str(vision.get("predicted_fault") or "visual anomaly").replace("_", " ").lower()

    def result_text(item: dict[str, Any]) -> str:
        return " ".join(
            [
                str(item.get("document_id", "")),
                str(item.get("title", "")),
                str(item.get("section", "")),
                " ".join(str(tag) for tag in item.get("tags", [])),
                str(item.get("text", "")),
            ]
        ).lower()

    relevant_results = [
        item
        for item in results
        if fault != "visual anomaly" and all(word in result_text(item) for word in fault.split())
    ]
    if not relevant_results:
        relevant_results = [
            item
            for item in results
            if any(token in result_text(item) for token in ("sop", "guide", "inspection", "maintenance", "leak", "lubrication", "seal", "corrosion", "crack"))
        ]
    if not relevant_results:
        relevant_results = results[:3]
    citations = [item.get("document_id") for item in relevant_results[:3] if item.get("document_id")]

    if top_score < 0.08:
        return {
            "answer": "Out of knowledge. I could not find enough matching telemetry, vision, or maintenance knowledge to answer that safely.",
            "confidence": 0.0,
            "citations": [],
            "retrieval_score": top_score,
            "mode": "out_of_knowledge",
        }

    rul = telemetry.get("predicted_rul")
    risk = telemetry.get("failure_risk")
    severity = telemetry.get("severity")
    score = vision.get("anomaly_score")

    cycle_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:cycles?|usage|uses?)\s*(?:per|/)\s*day", question_lower)
    if rul is not None and cycle_match:
        cycles_per_day = float(cycle_match.group(1))
        if cycles_per_day > 0:
            days = float(rul) / cycles_per_day
            return {
                "answer": (
                    f"About {float(rul):.1f} operating cycles remaining means roughly {days:.1f} days "
                    f"at {cycles_per_day:g} cycles per day. Because the image indicates {fault}, plan inspection before that limit instead of waiting for the full calculated life."
                ),
                "confidence": round(min(0.92, 0.55 + top_score / 2), 4),
                "citations": citations,
                "retrieval_score": top_score,
                "mode": "deterministic_cycle_math",
            }

    asks_next_step = any(phrase in question_lower for phrase in ("what to do", "next", "action", "recommend", "maintenance", "repair", "inspect"))
    if asks_next_step:
        guidance = str((relevant_results[0] if relevant_results else results[0]).get("text") or "").strip()
        actions = []
        if "oil leak" in fault or "leak" in fault:
            actions = [
                "Inspect the seal, fittings, drain points, and lubricant level.",
                "Clean the wet area and re-check after a short controlled run to confirm leak growth.",
                "Check for contamination and monitor bearing or gear wear because lubrication loss can accelerate damage.",
                "Keep telemetry under watch even though current failure risk is low.",
            ]
        elif "corrosion" in fault:
            actions = [
                "Clean the affected surface and inspect for coating loss, pitting, and material loss.",
                "Check moisture or chemical exposure sources.",
                "Apply anti-corrosion treatment and schedule a follow-up inspection.",
            ]
        elif "crack" in fault:
            actions = [
                "Remove the asset from high-load service until the indication is confirmed.",
                "Perform close visual inspection and nondestructive testing.",
                "Do not treat localization alone as proof of a crack.",
            ]
        else:
            actions = [
                "Inspect the highlighted area manually before assigning a final fault class.",
                "Compare with a clean reference image and continue telemetry monitoring.",
            ]
        return {
            "answer": (
                f"Next step: {actions[0]} "
                f"Then {actions[1].lower()} "
                f"Also {actions[2].lower() if len(actions) > 2 else 'document the finding and re-run inspection after correction.'} "
                f"Current telemetry is {severity} with about {float(rul):.1f} cycles remaining and {float(risk) * 100:.2f}% failure risk."
            ),
            "confidence": round(min(0.95, 0.45 + top_score), 4),
            "citations": citations,
            "retrieval_score": top_score,
            "mode": "deterministic_next_steps",
        }

    asks_rul = any(word in question_lower for word in ("cycle", "rul", "remaining", "life", "day", "days"))
    if asks_rul and rul is not None:
        return {
            "answer": (
                f"The telemetry result means about {float(rul):.1f} operating cycles remain before the learned failure threshold. "
                "A cycle is a unit of operation from the telemetry dataset, not automatically one clock day. "
                "To convert it to days, divide by your actual cycles per day."
            ),
            "confidence": round(min(0.9, 0.5 + top_score / 2), 4),
            "citations": citations,
            "retrieval_score": top_score,
            "mode": "deterministic_rul_explanation",
        }

    if not any(token in question_lower for token in fault.split() + ["telemetry", "vision", "failure", "risk", "cycle", "rul", "inspection", "maintenance", "part", "damage", "cause", "why"]):
        return {
            "answer": "Out of knowledge. I could not find enough matching telemetry, vision, or maintenance knowledge to answer that safely.",
            "confidence": 0.0,
            "citations": [],
            "retrieval_score": top_score,
            "mode": "out_of_knowledge",
        }

    parts = []
    if rul is not None:
        parts.append(f"The telemetry model estimates about {float(rul):.1f} operating cycles remaining.")
    if risk is not None:
        parts.append(f"The failure risk is {float(risk) * 100:.2f}% and telemetry status is {severity}.")
    if score is not None:
        parts.append(f"The image analysis found {fault} evidence with a visual score of {float(score) * 100:.1f}%.")
    if relevant_results:
        guidance = str(relevant_results[0].get("text") or "").strip()
        if guidance:
            parts.append(f"Matched guidance: {guidance[:280]}")

    return {
        "answer": " ".join(parts) if parts else "The retrieved knowledge supports the current RCA, but no concise explanation could be formed.",
        "confidence": round(min(0.95, 0.35 + top_score), 4),
        "citations": citations,
        "retrieval_score": top_score,
        "mode": "deterministic_rag_chat",
    }


def chat_prompt(question: str, context: dict[str, Any], rag: dict[str, Any]) -> list[dict[str, str]]:
    compact_rag = [
        {
            "document_id": item.get("document_id"),
            "title": item.get("title"),
            "text": item.get("text"),
            "score": item.get("score"),
        }
        for item in (rag.get("results") or [])[:5]
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are an industrial maintenance assistant. Answer in clear human language. "
                "Use only the supplied telemetry, DINO/vision, and retrieved knowledge. "
                "If the answer is not supported by retrieved knowledge, say exactly: Out of knowledge."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "telemetry": context.get("telemetry"),
                    "vision": context.get("vision"),
                    "retrieved_knowledge": compact_rag,
                    "required_json_schema": {
                        "answer": "human readable answer or Out of knowledge",
                        "confidence": "number 0..1",
                        "citations": "array of document_id strings from retrieved_knowledge only",
                    },
                },
                indent=2,
            ),
        },
    ]


def generate_chat_answer(question: str, context: dict[str, Any], rag: dict[str, Any]) -> dict[str, Any]:
    fallback = fallback_chat_answer(question, context, rag)
    if fallback["mode"] in {
        "out_of_knowledge",
        "deterministic_cycle_math",
        "deterministic_next_steps",
        "deterministic_rul_explanation",
    }:
        return {**fallback, "rag": rag}

    try:
        llm_result = call_huggingface(chat_prompt(question, context, rag))
        answer = str(llm_result.get("answer") or "").strip()
        if not answer or answer.lower() == "out of knowledge":
            return {
                "answer": "Out of knowledge.",
                "confidence": 0.0,
                "citations": [],
                "retrieval_score": fallback["retrieval_score"],
                "mode": "out_of_knowledge",
                "rag": rag,
            }
        citations = [
            citation
            for citation in llm_result.get("citations", [])
            if citation in {item.get("document_id") for item in (rag.get("results") or [])}
        ]
        return {
            "answer": answer,
            "confidence": float(llm_result.get("confidence", fallback["confidence"])),
            "citations": citations or fallback["citations"],
            "retrieval_score": fallback["retrieval_score"],
            "mode": "huggingface_rag_chat",
            "rag": rag,
        }
    except Exception as exc:
        return {
            **fallback,
            "llm_error": f"{type(exc).__name__}: {exc}",
            "rag": rag,
        }
