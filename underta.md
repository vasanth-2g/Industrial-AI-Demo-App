# Session Handoff

This file is the compact handoff for the next session.
It records what is already built, where the data and artifacts live, how to run the app, and what still matters.

## 1. Current Goal

We are building an end-to-end industrial AI demo with:

1. Telemetry input
2. Vision input
3. Retrieval over maintenance knowledge
4. Rule-based / pretrained reasoning
5. A small local web app for demo execution

Important:

- LLM fine-tuning is deferred.
- The baseline must work end to end first.
- We are using the local demo package for runtime loading.

## 2. What Is Already Done

### Telemetry

- CMAPSS telemetry notebooks exist.
- A pooled XGBoost telemetry path was created and saved.
- Current telemetry artifacts are stored in the demo package.

Useful files:

- `telemetrics/note1_data_extraction.ipynb`
- `telemetrics/note2_fd001_rul_baseline.ipynb`
- `telemetrics/note3_fd001_risk_anomaly.ipynb`
- `note7_cmapss_pooled_xgboost.ipynb`
- `note7_telemetry_xgboost_end_to_end.ipynb`

Saved telemetry artifacts expected by the app:

- `telemetry_risk/xgb_rul_model.joblib`
- `telemetry_risk/xgb_risk_model.joblib`
- `telemetry_risk/feature_columns.json`
- `telemetry_risk/metadata.json`
- `telemetry_risk/telemetry_contract_example.json`

### Vision

- Vision baseline is DINOv2-style patch-memory anomaly localization.
- Synthetic mechanical defect flow exists.
- VisA preparation notebooks also exist.

Useful files:

- `vision/note1_visa_data_preparation.ipynb`
- `vision/note2_visa_dinov2_anomaly_localization.ipynb`
- `vision/note3_synthetic_mechanical_defects.ipynb`
- `VisA_data_preparation.ipynb`
- `synthetic_mechanical_dinov2.ipynb`
- `note4_synthetic_mechanical_dinov2.ipynb`

Saved vision artifacts expected by the app:

- `vision_dinov2/normal_patch_memory.pt`
- `vision_dinov2/metadata.json`
- `vision_dinov2/vision_contract_example.json`
- `vision_dinov2/test_results.csv`
- `vision_dinov2/figures/defect_heatmaps.png`

### Knowledge / RAG

- Maintenance knowledge RAG notebook exists.
- The demo package should contain the RAG artifacts.

Useful file:

- `knowledge_rag.ipynb`

Expected artifacts:

- `knowledge_rag/rag_contract_example.json`
- `knowledge_rag/retrieval_evaluation.csv`

### Multimodal RCA

- Fusion / RCA flow exists.
- The app backend combines telemetry, vision, and RCA outputs.

Useful file:

- `note6_multimodal_fusion_rule_rca.ipynb`

Expected artifacts:

- `multimodal_rca/rule_rca_output.json`
- `multimodal_rca/normalized_evidence.json`
- `multimodal_rca/execution_trace.json`
- `multimodal_rca/reasoning_input.json`

### App

The working app lives under:

- `app/backend/server.py`
- `app/backend/scenario_runtime.py`
- `app/backend/vision_runtime.py`
- `app/backend/run_vision_rca.py`
- `app/frontend/index.html`

The same app folder was mirrored into the demo package workspace too.

## 3. Runtime Layout

### Repo Layout

```text
Hackathon_TCS_AMD_MULTIMODAL_004/
  app/
    backend/
      server.py
      scenario_runtime.py
      vision_runtime.py
      run_vision_rca.py
    frontend/
      index.html
  docs/
  telemetrics/
  vision/
  scripts/
```

### Demo Package Layout

The app server runs end-to-end inference from files inside the app folder:

```text
app/backend/XGboost/telemetry_risk
app/runtime
```

Expected structure there:

```text
app/
  backend/
    XGboost/
      telemetry_risk/
  frontend/
  runtime/
```

### App Folder Boundary

Treat `app/` as the intended standalone application folder.

Inside `app/`:

- `app/backend/`
- `app/frontend/`

The current backend does not directly import the old repo-level notebooks.
Runtime prediction uses trained model/artifact files copied inside `app/`.
Request inputs and generated outputs are stored under `app/runtime`.

The model/artifact files inside `app/` were produced earlier by notebooks and
pipelines in the repo-level work folders:

- `data/`
- `vision/`
- `telemetrics/`
- `knowledge/`

In other words: those folders are connected to the app through generated model
and metadata files, not through direct runtime imports. The notebooks create or
prepare the trained model outputs and metadata; the backend processes new
frontend inputs at runtime.

Current app-local runtime connections:

- `app/backend/XGboost/telemetry_risk`
- `app/runtime`

Files expected from that package:

- `telemetry_risk/metadata.json`
- `telemetry_risk/telemetry_contract_example.json`
- `telemetry_risk/xgb_rul_model.joblib`
- `telemetry_risk/xgb_risk_model.joblib`
- `telemetry_risk/feature_columns.json`
- `vision_dinov2/vision_contract_example.json`
- `vision_dinov2/metadata.json`
- `vision_dinov2/figures/defect_heatmaps.png`
- `knowledge_rag/rag_contract_example.json`
- `knowledge_rag/retrieval_evaluation.csv`
- `multimodal_rca/rule_rca_output.json`
- `multimodal_rca/normalized_evidence.json`
- `multimodal_rca/execution_trace.json`
- `multimodal_rca/reasoning_input.json`

Environment variables connected to the app:

- `DEMO_HOST`
- `DEMO_PORT`

Repo-level folders that are not directly used by the current app runtime:

- `data/`
- `knowledge/`
- `telemetrics/`
- `vision/`
- root notebooks such as `note4_synthetic_mechanical_dinov2.ipynb`,
  `note6_multimodal_fusion_rule_rca.ipynb`, and
  `note7_telemetry_xgboost_end_to_end.ipynb`

Note: `app/backend/XGboost/telemetry_risk` is the trained XGBoost model source
used for telemetry runtime prediction.

### Notebook Artifact Verification

Verified by reading the `.ipynb` files:

- `telemetrics/rul_baseline.ipynb` creates baseline C-MAPSS RUL artifacts under
  `artifacts/cmapss/fd001/`, including `metadata.json`,
  `official_test_predictions.csv`, `feature_importance.csv`, and
  `xgboost_fd001_rul.json`.
- `telemetrics/risk_baseline.ipynb` creates telemetry risk artifacts under
  `artifacts/cmapss/fd001/risk/`, including `metadata.json`,
  `official_test_risk_outputs.csv`, `telemetry_contract_example.json`, and
  `xgboost_fd001_failure_classifier.json`.
- `note7_telemetry_xgboost_end_to_end.ipynb` creates the newer joblib-based
  telemetry outputs under `artifacts/telemetry_xgboost/`, including
  `xgb_rul_model.joblib`, `xgb_risk_model.joblib`, `feature_columns.json`,
  `test_results.csv`, and `telemetry_contract_example.json`.
- `note4_synthetic_mechanical_dinov2.ipynb` creates vision artifacts under
  `/workspace/notebooks/vision/artifacts/synthetic_mechanical/dinov2`,
  including `normal_patch_memory.pt`, `test_results.csv`, `metadata.json`,
  `vision_contract_example.json`, and `figures/defect_heatmaps.png`.
- `knowledge_rag.ipynb` creates RAG artifacts under
  `/workspace/notebooks/knowledge/artifacts/rag`, including
  `maintenance_embeddings.npy`, `source_store.jsonl`, `metadata.json`,
  `rag_contract_example.json`, and `retrieval_evaluation.csv`.
- `note6_multimodal_fusion_rule_rca.ipynb` and
  `multimodal_fusion_rule_rca.ipynb` create RCA/fusion artifacts under
  `/workspace/notebooks/integration/artifacts/multimodal_rule_rca`, including
  `normalized_evidence.json`, `rule_rca_output.json`,
  `reasoning_input.json`, and `execution_trace.json`.

Conclusion: the trained models and metadata are predefined outputs from
notebooks, but telemetry and vision outputs are generated fresh from frontend
inputs at runtime.

## 4. Data Locations

### Required Data

#### Telemetry

CMAPSS files should be available under one of these:

- `data/CMAPSSData/`
- `notebooks/data/CMAPSSData/`
- any path used by the telemetry notebook if it resolves correctly

Required files:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

#### Vision

For the current demo, the important image data is:

- VisA
- synthetic mechanical source images
- generated anomaly artifacts

Typical locations used in the work:

- `data/VisA/`
- `data/synthetic_mechanical/`
- `notebooks/data/synthetic_mechanical/`

#### Knowledge

Knowledge inputs are stored as synthetic maintenance documents and JSONL/RAG artifacts.

Useful location:

- `notebooks/data/knowledge/maintenance_knowledge.jsonl`

### Data That Can Be Left Out of Git

These are heavy or generated and were treated as non-git assets:

- `data/VisA/`
- `data/CMAPSSData/`
- `data/_cmapss_extract/`
- generated vision artifacts
- other large raw downloads

## 5. What To Train

### Telemetry Model

Train or load the XGBoost telemetry model for:

- RUL prediction
- failure risk prediction

Outputs expected by the app:

- `predicted_rul`
- `failure_risk`
- `risk_label`
- `severity`

### Vision Model

Use the DINOv2 patch-memory anomaly path for:

- anomaly score
- heatmap
- approximate defect localization

Current state:

- baseline anomaly detection is working
- it is not yet a reliable defect classifier

### RCA Layer

The RCA step should fuse:

- telemetry output
- vision output
- retrieval evidence

The current demo is rule-based / deterministic fallback driven.

## 6. How To Run The Server

From the repo root:

```bash
python app/backend/server.py
```

Default dashboard:

```text
http://127.0.0.1:8000
```

The backend expects files under:

- `telemetry_risk`
- `vision_dinov2`
- `knowledge_rag`
- `multimodal_rca`

## 7. API Surface

Current endpoints:

- `GET /api/health`
- `GET /api/demo`
- `POST /api/infer`
- `GET /api/file/heatmap`
- `GET /api/file/rag_metrics`
- `GET /api/file/reasoning_input`

Important:

- `/api/infer` is POST only.
- Opening it directly in a browser with GET will return 404.
- `/api/infer` accepts JSON or `multipart/form-data`.
- For multipart input, send `scenario` as a JSON string and `image` as the
  uploaded inspection image file.
- Uploaded images are stored under `app/runtime/runs/<run>/inputs`.
- Generated outputs and heatmaps are stored under
  `app/runtime/runs/<run>/outputs`.
- `app/runtime` is deleted and recreated on server start.
- Live XGBoost telemetry scoring uses the `.joblib` files in
  `app/backend/XGboost/telemetry_risk`.

## 8. What The User Can Send

The app is designed for scenario-style input.

Telemetry can be:

- a scenario JSON
- telemetry history rows
- a structured payload with sensor values

Vision can be:

- an image path
- a packaged contract example

The end goal is:

```json
{
  "scenario_id": "SCN-001",
  "asset_id": "ROBOT-JOINT-01",
  "telemetry": { ... },
  "image_path": "path/to/image.jpg"
}
```

and the system returns:

- telemetry result
- vision result
- fused RCA result
- recommended actions

## 9. Local Git Notes

### Current repo state

Use `git status` before making changes.

### Safe workflow

1. Edit only the needed files.
2. Verify the app still runs.
3. Commit after the demo path is stable.

### If remote push fails

The earlier remote issues were caused by:

- wrong GitHub account
- repository ownership mismatch
- remote URL mismatch

Useful checks:

```bash
git remote -v
git branch -vv
git status
```

If credentials are stale, re-login with Git Credential Manager.

## 10. Current Warning / Deferred Work

Do not spend time on these before the baseline is stable:

- LLM fine-tuning
- extra new datasets
- over-complicated vision classifier changes
- replacing the app with a new architecture

The next useful step is to make the current telemetry + vision + RAG + RCA path reliable and repeatable.

## 11. Suggested Next Session Order

1. Confirm the telemetry artifacts are present and load correctly.
2. Confirm the vision artifacts are present and heatmap rendering works.
3. Confirm RAG artifacts are present and readable.
4. Run `python app/backend/server.py`.
5. Send a scenario through `POST /api/infer`.
6. Check the UI and returned JSON.
7. Only after the baseline is stable, consider classifier improvements.

## 12. Short Checklist

- [ ] Telemetry models load from `telemetry_risk`
- [ ] Vision artifacts load from `vision_dinov2`
- [ ] RAG artifacts load from `knowledge_rag`
- [ ] RCA output returns valid JSON
- [ ] App serves from `app/backend/server.py`
- [ ] `/api/infer` works with POST
- [ ] Demo response is reproducible
