# Industrial AI Maintenance Copilot Overview

This app is self-contained inside `app/`. It accepts a scenario text file and an inspection image, runs telemetry prediction, vision anomaly detection, RAG retrieval, and RCA generation, then displays the result in the frontend.

## Folder Layout

```text
app/
  frontend/
    index.html                 Main inference UI
    rag.html                   RAG upload and work-order search UI
  backend/
    server.py                  HTTP server and API endpoints
    scenario_runtime.py        Telemetry XGBoost prediction + orchestration
    vision_runtime.py          DINOv2 image anomaly detection + heatmap
    rag_runtime.py             RAG parsing, indexing, retrieval, work-order search
    llm_rca_runtime.py         Hugging Face RCA reasoning + fallback RCA
    model_config.py            Reads app/model_config.json
    XGboost/telemetry_risk/    Trained XGBoost telemetry models
    vision_dinov2/             DINOv2 model, metadata, patch memory
    knowledge_rag/             RAG documents, vectors, work-order index
  model_config.json            Manual model path/config file
  model_download_and_swap.ipynb Notebook for downloading/swapping models
  runtime/                     Generated per-run input/output files
```

## Frontend Pages

### Main Inference Page

```text
app/frontend/index.html
URL: http://127.0.0.1:8000/
```

Inputs:

- `input.txt`: scenario JSON or plain scenario text
- image file: inspection image such as crack, corrosion, oil leak, normal, etc.

Displays:

- RCA Status
- Predicted RUL
- Failure Risk
- RCA Confidence
- Telemetry evidence
- Vision DINOv2 heatmap/result image
- Root cause analysis
- Retrieved RAG citations
- Safety/limitations
- Raw RCA JSON

### RAG Documents Page

```text
app/frontend/rag.html
URL: http://127.0.0.1:8000/rag.html
```

Inputs:

- SOP text files
- PDFs
- Excel work orders / spare parts sheets
- HTML files
- SVG/DXF drawing text
- inspection/reference images

Displays:

- RAG document count
- uploaded chunk count
- visual index count
- raw RAG index status
- work-order parts/damage search results

## Backend APIs

### Health

```text
GET /api/health
```

Returns server status and app paths.

### App Info

```text
GET /api/demo
```

Returns configured app/model folders.

### Inference

```text
POST /api/infer
```

Accepted formats:

- `multipart/form-data`
  - `input`: uploaded `input.txt`
  - `image`: uploaded inspection image
- JSON body for direct API calls

Main output:

```json
{
  "telemetry": {},
  "vision": {},
  "rag": {},
  "llm_rca": {},
  "fusion": {},
  "run": {}
}
```

### Runtime Files

```text
GET /api/runtime/<path>
```

Used to retrieve generated heatmaps/results from `app/runtime`.

### RAG Documents

```text
GET /api/rag/documents
POST /api/rag/documents
```

`GET` returns index status.

`POST` uploads and indexes RAG documents.

### Work Order Search

```text
GET /api/work-orders/search?q=<query>
POST /api/work-orders/search
```

Searches Excel work-order/spare-part rows for:

- damaged part
- damage type
- work order id
- corrective action
- report text
- asset id

## Input Types

### Scenario Input

The expected file is `input.txt`.

It can be plain text, but for full telemetry prediction it should be JSON.

Minimal JSON:

```json
{
  "scenario_id": "SCENARIO-001",
  "asset_id": "ENGINE-001",
  "asset_type": "Industrial Turbofan Engine",
  "description": "Inspection scenario text",
  "location": "output shaft seal",
  "operating_state": "continuous_operation"
}
```

XGBoost-ready JSON:

```json
{
  "scenario_id": "SCENARIO-001",
  "asset_id": "ENGINE-001",
  "asset_type": "Industrial Turbofan Engine",
  "subset": "FD004",
  "description": "Telemetry and image inspection scenario",
  "telemetry_history": [
    {
      "unit": 1,
      "cycle": 1,
      "op_setting_1": 20.0072,
      "op_setting_2": 0.7,
      "op_setting_3": 100.0,
      "sensor_1": 491.19,
      "sensor_2": 606.67,
      "sensor_3": 1481.04
    }
  ]
}
```

Important:

- XGBoost does not use summary values like `failure_risk` or `predicted_rul` as inputs.
- XGBoost predicts those values from `telemetry_history`.
- The required telemetry schema is C-MAPSS style:

```text
unit, cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1 ... sensor_21
```

### Image Input

Supported:

```text
.png, .jpg, .jpeg, .bmp, .webp, .tif, .tiff
```

Used by:

- DINOv2 anomaly scoring
- heatmap/result image generation
- visual RAG matching against image SOPs/reference images

### RAG Document Inputs

Supported:

```text
.txt, .md, .json, .jsonl, .csv, .log
.pdf
.xlsx, .xls
.html, .htm
.svg, .dxf
.png, .jpg, .jpeg, .bmp, .webp, .tif, .tiff
```

## Execution Flow

### 1. Frontend Upload

The browser sends:

```text
input.txt + image
```

to:

```text
POST /api/infer
```

### 2. Server Creates A Run Folder

`server.py` creates:

```text
app/runtime/runs/<timestamp>/
  inputs/
    input.txt
    uploaded_image.png
    scenario.json
  outputs/
    result.json
    vision_dinov2_result.png
```

`app/runtime` is cleared when the server starts.

### 3. Telemetry Prediction

File:

```text
app/backend/scenario_runtime.py
```

Uses:

```text
app/backend/XGboost/telemetry_risk/xgb_rul_model.joblib
app/backend/XGboost/telemetry_risk/xgb_risk_model.joblib
app/backend/XGboost/telemetry_risk/feature_columns.json
```

Outputs:

- predicted RUL
- failure risk
- severity
- risk label
- top XGBoost features/sensors
- model evidence

If no `telemetry_history` is supplied:

- telemetry mode becomes `unavailable`
- no real XGBoost prediction is made

### 4. Vision Prediction

File:

```text
app/backend/vision_runtime.py
```

Uses:

```text
app/backend/vision_dinov2/facebook_dinov2_base
app/backend/vision_dinov2/normal_patch_memory.pt
app/backend/vision_dinov2/metadata.json
```

Outputs:

- anomaly score
- anomaly boolean
- predicted visual fault label
- DINOv2 heatmap/result image
- thresholds and evidence

Fault labels are inferred from image/context names such as:

```text
crack, corrosion, oil_leak, wear, overheating
```

### 5. RAG Retrieval

File:

```text
app/backend/rag_runtime.py
```

RAG indexes:

```text
app/backend/knowledge_rag/maintenance_knowledge.jsonl
app/backend/knowledge_rag/uploaded_documents.jsonl
app/backend/knowledge_rag/text_vector_index.jsonl
app/backend/knowledge_rag/visual_index.jsonl
app/backend/knowledge_rag/work_order_index.jsonl
```

Retrieval combines:

- persistent local text vectors
- TF-IDF semantic score
- keyword overlap
- code/tag boosts
- DINOv2 visual similarity for image documents

Outputs:

- top retrieved passages
- citations
- scores
- retrieval mode
- visual/text match counts

### 6. RCA Generation

File:

```text
app/backend/llm_rca_runtime.py
```

Inputs:

- telemetry output
- vision output
- RAG retrieval output

Uses:

- Hugging Face Transformers model if configured and downloaded
- deterministic fallback if model is not available

Outputs:

- classification / status
- confidence score
- root cause
- evidence summary
- next steps
- recommended actions
- citations
- safety and limitations

## Output Types

### Telemetry Output

```json
{
  "predicted_rul": 129.45,
  "failure_risk": 0.00013,
  "severity": "normal",
  "evidence": {
    "model_mode": "runtime_xgboost",
    "history_length": 120,
    "top_sensors": []
  }
}
```

### Vision Output

```json
{
  "anomaly_score": 0.797,
  "is_anomaly": true,
  "predicted_fault": "oil_leak",
  "heatmap_path": "app/runtime/runs/.../outputs/vision_dinov2_result.png",
  "heatmap_url": "/api/runtime/runs/.../outputs/vision_dinov2_result.png"
}
```

### RAG Output

```json
{
  "mode": "local_vector_tfidf_keyword_visual_hybrid",
  "results": [
    {
      "document_id": "USER-CMAPSS_FD004_README-001",
      "title": "NASA C-MAPSS Turbofan Degradation Readme",
      "score": 0.5
    }
  ]
}
```

### RCA Output

```json
{
  "status": "warning",
  "classification": "warning",
  "confidence": 0.59,
  "root_cause": "DINOv2 detected a localized visual anomaly; telemetry is normal.",
  "next_steps": [],
  "citations": []
}
```

## Retrieval And Display

1. Backend writes `result.json` under the current run output folder.
2. Backend returns the same JSON to the frontend.
3. Frontend renders telemetry, vision, RAG, and RCA sections.
4. Heatmap/result image is loaded from:

```text
/api/runtime/runs/<timestamp>/outputs/vision_dinov2_result.png
```

## Model Configuration

Manual model paths are controlled by:

```text
app/model_config.json
```

Use this file to change:

- XGBoost folder
- DINOv2 folder
- patch memory file
- Hugging Face LLM model/path
- RAG knowledge folder

Restart the server after editing.

## What Can Be Deleted

The app can run with only `app/` if the model folders inside `app` are kept.

Do not delete:

```text
app/backend/XGboost
app/backend/vision_dinov2
app/backend/knowledge_rag
app/backend/*.py
app/frontend
app/model_config.json
```

Safe to delete:

```text
app/runtime
app/backend/__pycache__
```

