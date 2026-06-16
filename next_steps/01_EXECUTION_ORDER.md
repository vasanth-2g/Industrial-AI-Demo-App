# Execution Order After Clone

Follow this order after cloning `notebooks_backup` on the Jupyter server.

## 0. Restore Ignored Datasets

Run:

```text
restore_ignored_datasets.ipynb
```

Expected outputs:

```text
data/CMAPSSData/
data/_cmapss_extract/
data/VisA/
data/dataset_restore_report.json
```

Pass condition:

```text
C-MAPSS validation passed.
VisA validation passed.
Dataset restore complete.
```

## 1. Telemetry Data Check

Run:

```text
telemetrics/Telemetries_data_extraction.ipynb
```

Expected:

```text
All FD001-FD004 files exist.
FD001 sanity check passes.
```

## 2. RUL Baseline

Run:

```text
telemetrics/rul_baseline.ipynb
```

Run through:

```text
13. Save Reproducible Artifacts
```

Expected outputs:

```text
telemetrics/artifacts/cmapss/fd001/xgboost_fd001_rul.json
telemetrics/artifacts/cmapss/fd001/metadata.json
telemetrics/artifacts/cmapss/fd001/official_test_predictions.csv
```

## 3. Failure Risk Baseline

Run:

```text
telemetrics/risk_baseline.ipynb
```

Run through:

```text
10. Save Artifacts
```

Expected output required by fusion:

```text
telemetrics/artifacts/cmapss/fd001/risk/telemetry_contract_example.json
```

## 4. VisA Dataset Manifest

Run:

```text
vision/btad_data_preparation.ipynb
```

Run through:

```text
6. Save Manifest and Dataset Report
```

Expected outputs:

```text
vision/artifacts/visa/visa_manifest.csv
vision/artifacts/visa/visa_validation_report.json
```

## 5. VisA DINOv2 Baseline

Run:

```text
vision/patchD_load_ROCm.ipynb
```

Run through:

```text
12. Save Memory Bank, Results, and Vision Contract
```

Expected outputs:

```text
vision/artifacts/visa/dinov2/pcb1/metadata.json
vision/artifacts/visa/dinov2/pcb1/vision_contract_example.json
```

This validates the DINOv2 architecture on VisA.

## 6. Synthetic Mechanical Defect Dataset

Before running, make sure this folder has clean normal mechanical images:

```text
data/synthetic_mechanical/source_normal/
```

Run:

```text
vision/synthetic_mechanical_defects.ipynb
```

Run through:

```text
4. Generate Images, Masks, and Manifest
5. Validate Generated Images and Masks
7. Save Dataset Report
```

Expected:

```text
invalid_samples: 0
```

Expected outputs:

```text
data/synthetic_mechanical/generated/manifest.csv
data/synthetic_mechanical/generated/generation_report.json
```

## 7. Synthetic Mechanical DINOv2 Demo

Run:

```text
vision/synthetic_mechanical_dinov2.ipynb
```

Run through:

```text
11. Save Artifacts and Vision Contract
```

Expected output required by fusion:

```text
vision/artifacts/synthetic_mechanical/dinov2/vision_contract_example.json
```

Contract must say:

```json
"predicted_fault": "visual_anomaly"
```

## 8. Knowledge and RAG

Run:

```text
knowledge_rag.ipynb
```

Run all cells.

Expected output required by fusion:

```text
knowledge/artifacts/rag/rag_contract_example.json
```

Expected metrics:

```text
Recall@1: 1.0
Recall@3: 1.0
Recall@5: 1.0
MRR:      1.0
```

## 9. Multimodal Fusion and Rule RCA

Run:

```text
integration/note6_multimodal_fusion_rule_rca.ipynb
```

If the notebook is not in the repo yet, copy it from the parent project or add
it before running.

Run all cells.

Expected outputs:

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

## 10. Optional Later Work

Only after the base end-to-end run works:

```text
1. Add more normal mechanical source images.
2. Re-run synthetic mechanical generation.
3. Improve calibrated vision anomaly confidence.
4. Tune XGBoost hyperparameters.
5. Add pretrained reasoning model call.
6. Consider LLM fine-tuning only after measured baseline failures.
```

