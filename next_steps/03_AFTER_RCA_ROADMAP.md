# After RCA Roadmap

After `integration/note6_multimodal_fusion_rule_rca.ipynb` runs successfully,
the baseline intelligence stack is complete:

```text
Telemetry + Vision + RAG -> Evidence Fusion -> Rule RCA
```

The next work is demo packaging, dashboard, optional reasoning model, and final
evaluation.

## 1. Confirm RCA Outputs

Run:

```text
integration/note6_multimodal_fusion_rule_rca.ipynb
```

Required output files:

```text
integration/artifacts/multimodal_rule_rca/normalized_evidence.json
integration/artifacts/multimodal_rule_rca/rule_rca_output.json
integration/artifacts/multimodal_rule_rca/reasoning_input.json
integration/artifacts/multimodal_rule_rca/execution_trace.json
```

Pass condition:

```text
RCA schema validation: PASSED
Citation validation: PASSED
Safety validation: PASSED
```

## 2. Create Final Demo Package

Collect these files into one demo folder:

```text
demo_artifacts/
  telemetry_contract_example.json
  vision_contract_example.json
  rag_contract_example.json
  normalized_evidence.json
  rule_rca_output.json
  reasoning_input.json
  execution_trace.json
  defect_heatmaps.png
  retrieval_evaluation.csv
  risk_metrics.json or metadata.json
  vision_metadata.json
```

Suggested source paths:

```text
telemetrics/artifacts/cmapss/fd001/risk/telemetry_contract_example.json
vision/artifacts/synthetic_mechanical/dinov2/vision_contract_example.json
knowledge/artifacts/rag/rag_contract_example.json
integration/artifacts/multimodal_rule_rca/*.json
vision/artifacts/synthetic_mechanical/dinov2/figures/defect_heatmaps.png
knowledge/artifacts/rag/retrieval_evaluation.csv
```

Purpose:

```text
One folder should prove the complete baseline can run end to end.
```

## 3. Build Simple Dashboard

Recommended tool:

```text
Streamlit
```

Dashboard layout:

```text
Left panel:
  - asset ID
  - predicted RUL
  - failure risk
  - telemetry severity

Middle panel:
  - inspection image
  - DINOv2 heatmap / overlay
  - visual anomaly status

Right panel:
  - RCA status
  - root cause hypothesis
  - confidence
  - evidence
  - citations
  - recommended actions
  - limitations
```

Expected dashboard input:

```text
demo_artifacts/
```

Do not require model reruns inside the dashboard. Load saved JSON and images.

## 4. Optional Pretrained Reasoning Model

Use only after the deterministic RCA works.

Input file:

```text
integration/artifacts/multimodal_rule_rca/reasoning_input.json
```

Model role:

```text
Reasoning and explanation only.
```

The model must not:

```text
- replace XGBoost telemetry predictions
- replace DINOv2 anomaly detection
- invent crack/corrosion/leak classes from DINOv2 output
- cite documents not retrieved by RAG
- recommend autonomous machine action
```

Compare model output against:

```text
integration/artifacts/multimodal_rule_rca/rule_rca_output.json
```

Keep the rule-based RCA as fallback.

## 5. Demo Validation Cases

Prepare three scripted cases:

```text
Case 1: Critical multimodal anomaly
  telemetry: high risk / low RUL
  vision: visual_anomaly true
  expected: critical status, inspect and safe-state recommendation

Case 2: Telemetry-only risk
  telemetry: high risk / low RUL
  vision: unavailable or anomaly false
  expected: warning or critical, lower confidence, inspect telemetry causes

Case 3: Vision-only anomaly
  telemetry: low risk
  vision: visual_anomaly true
  expected: warning, inspect localized region, do not claim failure
```

Validation checks:

```text
- Output JSON is valid.
- Citations are from retrieved documents only.
- Safety field blocks autonomous control.
- Missing/conflicting modalities lower confidence.
- Limitations are visible.
```

## 6. Final Demo Story

Use this explanation:

```text
The system combines telemetry prognosis, visual anomaly localization, and
maintenance retrieval. It produces a structured, cited RCA recommendation with
visible uncertainty and safety guardrails. The LLM is optional; the baseline
already works with deterministic RCA.
```

Important wording:

```text
DINOv2 detects and localizes visual anomalies.
It does not reliably classify crack/corrosion/leak type in the current dataset.
```

## 7. Optional Improvements If Time Remains

Priority order:

```text
1. Add more independent normal mechanical source images.
2. Re-run synthetic mechanical generation.
3. Improve calibrated visual anomaly confidence.
4. Tune XGBoost hyperparameters.
5. Add pretrained Qwen/Instruct reasoning.
6. Add Streamlit styling and screenshots.
7. Fine-tune only after measured baseline failures.
```

Do not combine C-MAPSS rows and synthetic image features into one XGBoost
training table. They are different datasets. Fuse outputs at the RCA/demo layer.

