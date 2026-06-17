# Industrial AI Demo App

This app runs end-to-end inference from frontend input.

Architecture overview:

```text
app/APP_OVERVIEW.md
```

The frontend sends:

- telemetry sensor data as scenario JSON
- an inspection image file

The backend generates fresh telemetry, vision, and fused RCA outputs for that
request.

## Run

From the repository root:

```bash
python app/backend/server.py
```

Open:

```text
http://127.0.0.1:8000
```

RAG document upload page:

```text
http://127.0.0.1:8000/rag.html
```

The same page also includes DINOv2 normal-reference memory upload. Use clean
normal asset images there to rebuild the patch memory used by visual anomaly
detection.

The backend uses trained model/artifact files inside `app`:

```text
app/backend/XGboost/telemetry_risk
app/backend/vision_dinov2
app/backend/knowledge_rag
```

Model download/swap notebook:

```text
app/model_download_and_swap.ipynb
```

Use it when you want to download a Hugging Face LLM, replace DINOv2, replace
XGBoost models, or rebuild RAG vectors.

If model weights are too large to upload to GitHub, clone the repo on the
server and download the vision weights there:

```bash
python app/download_dinov2_weights.py
```

This downloads:

```text
facebook/dinov2-base          -> app/backend/vision_dinov2/facebook_dinov2_base
google/owlvit-base-patch32    -> app/backend/vision_dinov2/google_owlvit_base_patch32
facebook/sam-vit-base         -> app/backend/vision_dinov2/facebook_sam_vit_base
```

Use `--only-dinov2` if you only need the current DINOv2 anomaly branch.

Robot ROI crop notebook:

```text
app/robot_roi_before_dinov2.ipynb
```

Use it to test cropping the robot/asset region before DINOv2 so plant
background does not dominate the visual anomaly overlay.

DINOv2 vision architecture:

```text
app/ARC_DINO.md
```

Manual model config file:

```text
app/model_config.json
```

Edit this file when you want to point the server at a different XGBoost folder,
DINOv2 folder, patch-memory file, or Hugging Face LLM path without changing
Python code. Restart the server after editing it.

Request inputs and generated outputs are written to `app/runtime/runs`.
`app/runtime` is deleted and recreated when the server starts, so old input
images and generated outputs are cleared on restart.

## API

```text
GET /api/health
GET /api/demo
GET /api/rag/documents
GET /api/vision/memory
GET /api/work-orders/search?q=<query>
POST /api/rag/documents
POST /api/vision/memory
POST /api/work-orders/search
POST /api/infer
GET /api/runtime/<generated-file>
```

`POST /api/infer` accepts a scenario JSON payload and returns telemetry,
vision, and fused RCA output.

`POST /api/infer` also accepts `multipart/form-data` with:

- `scenario`: JSON string containing telemetry sensor data
- `image`: uploaded image file

Uploaded images are stored under the current `app/runtime/runs/<run>/inputs`
folder. Generated results and heatmaps are stored under
`app/runtime/runs/<run>/outputs`.

Telemetry behavior:

- The backend loads trained XGBoost `.joblib` models from
  `app/backend/XGboost/telemetry_risk`.
- It returns predicted RUL, failure risk, risk label, severity, and evidence.

Vision behavior:

- The backend crops the uploaded image to the foreground asset/robot region
  before DINOv2 scoring when possible.
- The backend processes the cropped image and generates a marked defect image
  for that request.
- The generated heatmap is served through `/api/runtime/...`.
- `/api/vision/memory` rebuilds `normal_patch_memory.pt` from uploaded clean
  normal/reference images.

RAG and LLM RCA behavior:

- Upload SOPs or maintenance notes through `POST /api/rag/documents`.
- Text, PDF, spreadsheet, HTML, and SVG/DXF drawing-text files are chunked with
  overlap and indexed with persistent local text vectors, TF-IDF retrieval, and
  keyword/code boosting.
- Excel work-order and spare-part sheets are also indexed row-by-row into
  `app/backend/knowledge_rag/work_order_index.jsonl`. Search them with
  `/api/work-orders/search` to find damaged parts, damage type, corrective
  action, and report rows.
- Image files are stored, optionally OCR'd with `pytesseract` if available, and
  indexed with DINOv2 visual anomaly metadata.
- Image files also get lightweight layout analysis and DINOv2 global embeddings
  for visual document matching against the current inspection image.
- Hybrid retrieval combines persistent local text-vector similarity, TF-IDF text
  similarity, keyword/code boosting, and local NumPy cosine similarity over
  DINOv2 visual embeddings. The vector backend reports Pinecone/Chroma
  availability when those packages/configurations are present, otherwise it uses
  local JSONL vectors.
- `/api/infer` combines telemetry output, vision output, and hybrid RAG results.
- The backend uses a Hugging Face instruction model through `transformers` for
  structured RCA reasoning. The default model is
  `Qwen/Qwen2.5-0.5B-Instruct`; override it with `HF_LLM_MODEL`. If the model is
  not downloaded or cannot run, the backend returns a deterministic RCA with the
  same shape and records the Hugging Face error in `llm.error`.

Runtime XGBoost scoring requires `joblib`, `pandas`, `xgboost`, and
`scikit-learn`. The frontend is plain HTML/CSS/JavaScript.
