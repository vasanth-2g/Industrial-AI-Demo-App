For an **Industrial AI Maintenance Copilot – Hybrid RAG System**, here is a more detailed component list including **PDF readers, LLM models, vector databases, search engines, and deployment options**.

![Image](https://images.openai.com/static-rsc-4/4GHglpSpDtQLC837LWzSkZJUJ2WO9YY3KdKTyMBZyx_avLqAjfYYc9ZJMfivoC1CtQNXBMXxcuJQIYiD5fukrXxvsTRHhbPgz2YMhXf5nzS1qefZ0W1TCCBL84iGkogAq0z-abElsimSCiJI3EwM9P9mRR8hgpbSEoUCx4hx3A2pGZJ56Wnv9Vp-NlfAOs1s?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/V5e8fJJuC-cFtAwRsp9iqj0RRygxLIC7vMQuln7Rpkyzvs0IoEnNA3iSopFzc7GdgfoQ2zq61sIw_CA9_VmW-ukGD-cT-MbI6sjf0erxq96XY-CHZ-oih6dlYSebi0NpJnesspmIOUlJHngWegX31d9pT1hi-9aQp_uiEKiwNEeZec69LFetYkb6LWr_qRMG?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/9S6WfWqwOE-gsYeZm7ma-eZCweDpu4HpJELk5mSIIZZYrliscXYhnQfOYBzm_MXPk7U9dxbX8tXPM6IizTPkggsdhEzsqSXJD7pwwbFh6tJFdADLx9ssEAvIOTrxlRql5Uq1_fUCrNYf5zrcuwYsS4OxMLW5bzgSmb3FZ6ETONqGMW__hhv-voguvqtfVrPp?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/CMUi5042zJt-fGECEJXPrhoOISEYJ29McHZUIOHcFWBCqIsp2sP_49t_TVcSSkzaycdNIYXG38vV9sXyS4RlraJBPKLlM7fgKfQzG4cgaT-K9wOQ4ZK5x5vRiLnulSLAtl5AcZG7DxZOwdjaPhjmM4o2azizjDtyQ3Iu5N0XdJkpNSU-QdmJDtoftxdcHSRs?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/k5bFL-lbEmruQMRw2mLWUFdwUzZA7YXpT5zB3cLnfvaemBQz0PcjX9ESIr9Rdm1JIaiQFZzV_s-c__jhnFL3S5wCNlRBdxhW1QiiyE-1-D-sVeUWwGEAdhV3HrNsPIB6peW9mPakc6agey1n8Xb7hhcgP62Je_pjT9MTfZq4wKdbfoiN5kGo37jZV03tGrKI?purpose=fullsize)

# Industrial AI Maintenance Copilot – Hybrid RAG Components

## 1. Data Source Layer (Maintenance Knowledge)

### Input Documents

| Data Type      | Examples                                     |
| -------------- | -------------------------------------------- |
| PDF Manuals    | Pump manuals, compressor manuals, OEM guides |
| SOP Documents  | Maintenance procedures, safety instructions  |
| Work Orders    | Previous breakdown records                   |
| Excel Files    | Failure history, spare parts list            |
| CAD / Drawings | Equipment diagrams                           |
| HTML Pages     | Internal knowledge articles                  |
| Logs           | SCADA alarms, machine events                 |

---

# 2. Document Reader / Parser Layer

Purpose:
Convert industrial documents into clean text.

## PDF Readers

### Option 1: Simple PDF Text Extraction

**Libraries:**

* PyMuPDF

  * Fast PDF extraction
  * Extract text, images, tables

* PDFMiner

  * Detailed PDF parsing
  * Good for technical manuals

Example:

```
Pump_Manual.pdf
        |
        ↓
PDF Reader
        |
        ↓
Extracted Text
```

---

### Option 2: Enterprise Document AI

For scanned manuals:

* Microsoft Azure Document Intelligence
* Amazon Web Services Textract
* Google Document AI

Handles:

✓ OCR
✓ Tables
✓ Images
✓ Handwritten notes

---

# 3. OCR Layer (Scanned Documents)

Used when PDF is image-based.

Tools:

* Tesseract OCR
* Azure OCR
* Google Vision OCR

Example:

```
Scanned Maintenance Sheet
          |
          ↓
OCR Engine
          |
          ↓
Searchable Text
```

---

# 4. Text Processing Layer

## Chunking

Break documents into smaller pieces.

Example:

Manual:

```
50 page Pump Manual
        |
        ↓
Chunks

Chunk 1:
Pump installation

Chunk 2:
Bearing failure

Chunk 3:
Lubrication procedure
```

Typical:

* Chunk size: 500–1500 tokens
* Overlap: 100–300 tokens

## Metadata Creation

Each chunk stores:

```json
{
 "equipment":"Compressor",
 "model":"XYZ-200",
 "document":"Service Manual",
 "page":45,
 "section":"Troubleshooting"
}
```

---

# 5. Embedding Model Layer

Converts text → vectors.

## Popular Models

### Cloud

**OpenAI**

* text-embedding-3-small
* text-embedding-3-large

Good for:

* Enterprise RAG
* High accuracy

### Open Source

**BGE Models**

* BGE-small
* BGE-base
* BGE-large

**E5 Models**

* multilingual-e5-large

Good for:

* On-prem deployment

Example:

```
"bearing temperature high"

        ↓

[0.023,0.332,0.876,...]
```

---

# 6. Vector Database Layer

Stores embeddings.

## Option 1: Enterprise Cloud

### Azure AI Search

Good for:

✓ Hybrid search
✓ Enterprise security
✓ Microsoft ecosystem

Architecture:

```
Vector Search
+
Keyword Search
+
Filters
```

---

## Option 2: Dedicated Vector DB

### Pinecone

Good for:

* Managed cloud
* Fast similarity search

### Weaviate

Features:

* Vector search
* Hybrid search
* Graph relationships

### Milvus

Good for:

* Large industrial datasets
* Millions of documents

### FAISS

From Meta

Good for:

* Local prototype
* Research
* Small deployments

---

# 7. Keyword Search Engine

Required for Hybrid RAG.

## Options:

### Elasticsearch

Provides:

* BM25 search
* Exact keyword matching
* Filtering

Example:

User:

```
Alarm E101
```

Keyword search finds:

```
E101 Troubleshooting Guide
```

---

# 8. Hybrid Retriever

Combines:

```
Vector Search
       +
Keyword Search

       ↓

Combined Ranking
```

Example:

| Method  | Result                  |
| ------- | ----------------------- |
| Vector  | "motor vibration issue" |
| Keyword | "Alarm VIB-101"         |
| Hybrid  | Best match              |

---

# 9. Reranking Model

Improves search quality.

Options:

* BGE Reranker
* Cohere Reranker
* Cross Encoder models

Flow:

```
100 documents retrieved

          ↓

Reranker

          ↓

Top 5 documents
```

---

# 10. LLM Generation Layer

The LLM reads retrieved context and answers.

## Cloud LLMs

### OpenAI

Models:

* GPT-4.1
* GPT-4.1-mini
* GPT-5 family (where available)

Good for:

* Complex troubleshooting
* Reasoning

### Anthropic

Models:

* Claude family

Good for:

* Long documents

### Google

Models:

* Gemini family

---

## On-Prem LLMs

For factories with data restrictions:

### Llama

Models:

* Llama 3.x
* Llama 4 family

### Mistral AI

Models:

* Mixtral
* Mistral Large

Running:

* GPU server
* Kubernetes
* Private cloud

---

# 11. Prompt Layer

Controls answers.

Example:

```
You are an industrial maintenance expert.

Rules:
1. Use only provided documents
2. Mention source page
3. Do not guess

Question:
{query}

Context:
{documents}
```

---

# 12. Copilot Application Layer

Frontend:

* React
* Angular
* Teams app
* Web portal

Features:

✓ Chat
✓ Source citations
✓ Equipment lookup
✓ Maintenance recommendations
✓ Work order creation

---

# 13. Industrial Integration Layer

Connect:

| System    | Purpose         |
| --------- | --------------- |
| SCADA     | Live alarms     |
| Historian | Sensor trends   |
| CMMS      | Work orders     |
| ERP       | Spare parts     |
| MES       | Production data |

---

# Recommended Industrial Production Stack

A practical setup:

```
PDF/OCR
  |
PyMuPDF + Azure Document Intelligence
  |
Chunking
  |
OpenAI Embeddings
  |
Azure AI Search
  |
Hybrid Retrieval
  |
BGE Reranker
  |
GPT Model
  |
Maintenance Copilot UI
```

This stack fits an **industrial-grade RAG test SOP** because it supports traceability, citations, security, and evaluation.

---

# Implementation Checklist For This App

Status legend:

- [x] Complete in current `app`
- [~] Partial / prototype implementation
- [ ] Not implemented yet

## 1. Data Source Layer

- [x] Base maintenance knowledge stored in `app/backend/knowledge_rag/maintenance_knowledge.jsonl`
- [x] Uploaded SOP/document chunks stored in `app/backend/knowledge_rag/uploaded_documents.jsonl`
- [x] Uploaded files stored under `app/backend/knowledge_rag/uploads`
- [x] C-MAPSS readme can be uploaded as RAG knowledge
- [x] PDF manuals can be parsed with local PyMuPDF/`fitz`
- [x] Excel work orders / spare parts sheets can be parsed through pandas/openpyxl
- [x] SVG/DXF drawing text and metadata can be parsed for search
- [x] HTML pages can be parsed through BeautifulSoup

## 2. Document Reader / Parser Layer

- [x] Text-like files supported: `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, `.log`
- [x] PDF text extraction supported with PyMuPDF/`fitz`
- [x] Image documents supported: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tif`, `.tiff`
- [~] Image OCR is implemented as optional `pytesseract`, but `pytesseract` is not installed in the current environment
- [ ] Enterprise OCR / Document AI is not connected
- [~] Table extraction from PDFs is handled as text through PyMuPDF, not structured table cells

## 3. Text Processing Layer

- [x] Document chunking implemented in `rag_runtime.chunk_text`
- [x] Metadata stored per chunk: document id, title, type, asset type, revision, tags, source file
- [~] Chunking is character-based with overlap, not token-based
- [x] Chunk overlap is implemented
- [x] PDF chunks include page marker metadata when page markers are present

## 4. Embedding / Vector Layer

- [x] Image documents are converted to DINOv2 visual embeddings and stored in `visual_index.jsonl`
- [x] Query images are converted to DINOv2 embeddings for visual matching
- [x] Text retrieval uses persistent local hashing vectors plus TF-IDF at query time
- [x] Persistent local text embeddings are stored in `app/backend/knowledge_rag/text_vector_index.jsonl`
- [~] Hugging Face embedding packages are not downloaded; local hashing vectors are used without external model downloads

## 5. Vector Database / Search Layer

- [x] Local lightweight hybrid search implemented in `rag_runtime.retrieve`
- [x] Keyword overlap scoring implemented
- [x] TF-IDF semantic scoring implemented
- [x] DINOv2 visual cosine scoring implemented for image matches
- [~] Pinecone/ChromaDB backend detection is implemented; external packages/configuration are still required to activate those services
- [~] Elasticsearch/BM25 service is not integrated; local keyword scoring is implemented

## 6. Hybrid Retriever

- [x] Combines TF-IDF score, keyword score, code boost, tag boost, and visual similarity
- [x] Returns top-k RAG passages with scores and source metadata
- [~] Reranking is not a separate model; ranking is weighted scoring over TF-IDF, local vectors, keyword, code, tag, and visual scores
- [ ] BGE/Cohere/Cross-Encoder reranker is not implemented because it requires an additional model download/service

## 7. LLM RCA Layer

- [x] `llm_rca_runtime.py` consumes telemetry output, vision output, and RAG results
- [x] Output includes RCA status/classification, root cause, confidence score, evidence, next steps, recommended actions, citations, limitations, safety
- [x] Deterministic fallback works without an API key
- [x] Hugging Face Transformers local LLM path is connected for RCA reasoning
- [x] Default local model is `Qwen/Qwen2.5-0.5B-Instruct`, configurable with `HF_LLM_MODEL`
- [~] Model weights were not downloaded in this pass; deterministic fallback records the Hugging Face model error when weights are unavailable
- [ ] Larger industrial/on-prem model selection and GPU sizing are not finalized

## 8. Copilot Application Layer

- [x] Main inference page accepts only `input.txt` scenario and one image
- [x] RAG document upload page exists at `/rag.html`
- [x] RAG page supports document upload and index health display
- [x] RCA output displays telemetry, DINOv2 heatmap/result image, RAG citations, and raw JSON
- [~] UI is plain HTML/CSS/JS, not React/Angular
- [ ] Chat-style copilot interface is not implemented
- [ ] Work order creation is not implemented

## 9. Industrial Integration Layer

- [x] Offline C-MAPSS telemetry test data can be converted into scenario input
- [x] XGBoost telemetry prediction runs from `app/backend/XGboost/telemetry_risk`
- [x] DINOv2 visual anomaly detection runs from `app/backend/vision_dinov2`
- [ ] Live SCADA / historian / CMMS / ERP / MES integrations are not implemented

## Current End-To-End Flow Status

- [x] Upload `input.txt` scenario
- [x] Upload inspection image
- [x] Parse telemetry history from scenario
- [x] Run XGBoost RUL/risk prediction
- [x] Run DINOv2 visual anomaly detection
- [x] Generate DINOv2 heatmap/result image
- [x] Retrieve RAG context using telemetry + vision outputs
- [x] Include uploaded C-MAPSS readme/PDF knowledge in retrieval
- [x] Generate RCA JSON with status, confidence, evidence, next steps, and citations
- [x] Sensor-level XGBoost attribution is implemented with `pred_contribs` top features/top sensors
