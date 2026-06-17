from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RAG_ROOT = Path(__file__).resolve().parent / "knowledge_rag"
DOCUMENT_PATH = RAG_ROOT / "maintenance_knowledge.jsonl"
USER_DOCUMENT_PATH = RAG_ROOT / "uploaded_documents.jsonl"
UPLOAD_ROOT = RAG_ROOT / "uploads"
VISUAL_INDEX_PATH = RAG_ROOT / "visual_index.jsonl"
TEXT_VECTOR_INDEX_PATH = RAG_ROOT / "text_vector_index.jsonl"
WORK_ORDER_INDEX_PATH = RAG_ROOT / "work_order_index.jsonl"
TOP_K = 5
TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv", ".log"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
PDF_SUFFIXES = {".pdf"}
SPREADSHEET_SUFFIXES = {".xlsx", ".xls"}
HTML_SUFFIXES = {".html", ".htm"}
CAD_SUFFIXES = {".svg", ".dxf"}
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 180
LOCAL_VECTOR_FEATURES = 384
PART_KEYS = ("part", "component", "spare", "material", "item", "module", "assembly")
DAMAGE_KEYS = ("damage", "fault", "failure", "defect", "issue", "problem", "condition")
ACTION_KEYS = ("action", "repair", "recommendation", "maintenance", "resolution", "corrective")
ID_KEYS = ("work_order", "wo", "order", "ticket", "id", "case")


def load_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in (DOCUMENT_PATH, USER_DOCUMENT_PATH):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                documents.append(json.loads(line))
    return documents


def load_visual_index() -> list[dict[str, Any]]:
    if not VISUAL_INDEX_PATH.is_file():
        return []
    records = []
    for line in VISUAL_INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def append_visual_index(record: dict[str, Any]) -> None:
    RAG_ROOT.mkdir(parents=True, exist_ok=True)
    with VISUAL_INDEX_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_text_vector_index() -> list[dict[str, Any]]:
    if not TEXT_VECTOR_INDEX_PATH.is_file():
        return []
    records = []
    for line in TEXT_VECTOR_INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def local_text_embedding(text: str) -> list[float]:
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.preprocessing import normalize
    except Exception:
        tokens = tokenize(text)
        vector = np.zeros(LOCAL_VECTOR_FEATURES, dtype=np.float32)
        for token in tokens:
            vector[hash(token) % LOCAL_VECTOR_FEATURES] += 1.0
        norm = float(np.linalg.norm(vector))
        return (vector / norm).astype(float).tolist() if norm else vector.astype(float).tolist()

    vectorizer = HashingVectorizer(
        n_features=LOCAL_VECTOR_FEATURES,
        alternate_sign=False,
        norm=None,
        stop_words="english",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.transform([text])
    matrix = normalize(matrix)
    return matrix.toarray()[0].astype(float).tolist()


def append_text_vectors(documents: list[dict[str, Any]]) -> None:
    if not documents:
        return
    RAG_ROOT.mkdir(parents=True, exist_ok=True)
    with TEXT_VECTOR_INDEX_PATH.open("a", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {
                        "document_id": document["document_id"],
                        "source_file": document.get("source_file"),
                        "embedding_model": f"local_hashing_{LOCAL_VECTOR_FEATURES}",
                        "embedding": local_text_embedding(document_text(document)),
                    }
                )
                + "\n"
            )


def load_work_orders() -> list[dict[str, Any]]:
    if not WORK_ORDER_INDEX_PATH.is_file():
        return []
    records = []
    for line in WORK_ORDER_INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def append_work_orders(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    RAG_ROOT.mkdir(parents=True, exist_ok=True)
    with WORK_ORDER_INDEX_PATH.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_text_vector_index(documents: list[dict[str, Any]] | None = None) -> None:
    docs = documents if documents is not None else load_documents()
    RAG_ROOT.mkdir(parents=True, exist_ok=True)
    if TEXT_VECTOR_INDEX_PATH.exists():
        TEXT_VECTOR_INDEX_PATH.unlink()
    append_text_vectors(docs)


def normalize_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def pick_by_keys(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for column, value in row.items():
        normalized = normalize_header(column)
        if any(key in normalized for key in keys):
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none", "nat"}:
                return text
    return ""


def infer_work_order_records(filename: str, sheets: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    asset_type = str(metadata.get("asset_type") or "industrial_asset")
    for sheet_name, frame in sheets.items():
        clean_frame = frame.fillna("")
        for row_index, row in enumerate(clean_frame.to_dict(orient="records"), start=2):
            clean_row = {str(key): str(value).strip() for key, value in row.items() if str(value).strip()}
            if not clean_row:
                continue
            work_order_id = pick_by_keys(clean_row, ID_KEYS) or f"{Path(filename).stem}-{sheet_name}-{row_index}"
            part = pick_by_keys(clean_row, PART_KEYS)
            damage = pick_by_keys(clean_row, DAMAGE_KEYS)
            action = pick_by_keys(clean_row, ACTION_KEYS)
            report = "; ".join(f"{key}: {value}" for key, value in clean_row.items())
            record = {
                "record_id": f"WO-{re.sub(r'[^A-Za-z0-9]+', '_', str(work_order_id)).strip('_')[:48]}-{row_index}",
                "work_order_id": work_order_id,
                "asset_id": pick_by_keys(clean_row, ("asset", "equipment", "machine", "engine")),
                "asset_type": asset_type,
                "part": part,
                "damage_type": damage,
                "action": action,
                "report": report,
                "source_file": filename,
                "sheet": str(sheet_name),
                "row": row_index,
                "raw": clean_row,
                "tags": metadata.get("tags") or ["work_order", "spare_parts", "maintenance"],
            }
            records.append(record)
    return records


def search_work_orders(query: str, limit: int = 8) -> dict[str, Any]:
    started = time.perf_counter()
    records = load_work_orders()
    query_tokens = tokenize(query)
    query_embedding = local_text_embedding(query)
    scored = []
    for record in records:
        searchable = " ".join(
            str(record.get(key, ""))
            for key in ("work_order_id", "asset_id", "asset_type", "part", "damage_type", "action", "report")
        )
        text_score = keyword_score(query_tokens, tokenize(searchable))
        vector_score = cosine_similarity_list(query_embedding, local_text_embedding(searchable))
        part_boost = 0.15 if record.get("part") and str(record.get("part")).lower() in query.lower() else 0.0
        damage_boost = 0.15 if record.get("damage_type") and str(record.get("damage_type")).lower() in query.lower() else 0.0
        score = 0.45 * max(vector_score, 0.0) + 0.4 * text_score + part_boost + damage_boost
        scored.append({"record": record, "score": float(score), "keyword_score": float(text_score), "vector_score": float(vector_score)})

    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]
    results = [
        {
            **item["record"],
            "score": item["score"],
            "keyword_score": item["keyword_score"],
            "vector_score": item["vector_score"],
        }
        for item in ranked
        if item["score"] > 0 or query.strip() == ""
    ]
    damaged_parts: dict[str, int] = {}
    damage_types: dict[str, int] = {}
    for item in results:
        if item.get("part"):
            damaged_parts[str(item["part"])] = damaged_parts.get(str(item["part"]), 0) + 1
        if item.get("damage_type"):
            damage_types[str(item["damage_type"])] = damage_types.get(str(item["damage_type"]), 0) + 1

    return {
        "query": query,
        "mode": "work_order_spare_parts_local_vector_keyword",
        "record_count": len(records),
        "result_count": len(results),
        "damaged_parts": [
            {"part": part, "count": count}
            for part, count in sorted(damaged_parts.items(), key=lambda item: item[1], reverse=True)
        ],
        "damage_types": [
            {"damage_type": damage, "count": count}
            for damage, count in sorted(damage_types.items(), key=lambda item: item[1], reverse=True)
        ],
        "results": results,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }



def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap_chars: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [text.strip()]:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            overlap = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph
    if current:
        chunks.append(current)
    return chunks


def normalize_upload_document(filename: str, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    metadata = metadata or {}
    title = str(metadata.get("title") or Path(filename).stem)
    document_type = str(metadata.get("document_type") or "uploaded_sop")
    asset_type = str(metadata.get("asset_type") or "industrial_asset")
    revision = str(metadata.get("revision") or "uploaded")
    tags = metadata.get("tags")
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    if not isinstance(tags, list):
        tags = ["uploaded", "sop", "maintenance"]

    docs = []
    for index, chunk in enumerate(chunk_text(text), start=1):
        page_markers = [int(match) for match in re.findall(r"\[Page\s+(\d+)\]", chunk)]
        docs.append(
            {
                "document_id": f"USER-{Path(filename).stem.upper()[:24]}-{index:03d}",
                "title": title,
                "document_type": document_type,
                "asset_type": asset_type,
                "revision": revision,
                "effective_date": str(metadata.get("effective_date") or "uploaded"),
                "section": f"Uploaded section {index}",
                "chunk_index": index,
                "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
                "page_start": min(page_markers) if page_markers else metadata.get("page_start"),
                "page_end": max(page_markers) if page_markers else metadata.get("page_end"),
                "text": chunk,
                "tags": tags,
                "source_status": "uploaded",
                "source_file": filename,
            }
        )
    return docs


def analyze_image_layout(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None:
        return {"layout_type": "unknown", "regions": [], "confidence": 0.0}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    edges = cv2.Canny(gray, 80, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area >= max(80, 0.001 * width * height):
            boxes.append({"x": x, "y": y, "width": w, "height": h, "area": area})

    horizontal = sum(1 for box in boxes if box["width"] > box["height"] * 3)
    vertical = sum(1 for box in boxes if box["height"] > box["width"] * 3)
    large_regions = sum(1 for box in boxes if box["area"] > 0.08 * width * height)
    edge_density = float(edges.mean() / 255.0)
    color_std = float(np.asarray(image).std())

    if large_regions >= 1 and edge_density < 0.12:
        layout_type = "photo_or_diagram"
    elif horizontal >= 6 and vertical >= 3:
        layout_type = "table_or_form"
    elif vertical >= 4 and horizontal >= 2:
        layout_type = "graph_or_chart"
    elif horizontal >= 8:
        layout_type = "text_block"
    elif color_std > 45:
        layout_type = "mixed_visual_document"
    else:
        layout_type = "document_image"

    return {
        "layout_type": layout_type,
        "regions": boxes[:24],
        "edge_density": edge_density,
        "color_std": color_std,
        "confidence": 0.65 if boxes else 0.35,
    }


def add_uploaded_document(filename: str, content: bytes, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    RAG_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)[:120] or "document.txt"
    stored_path = UPLOAD_ROOT / safe_name
    stored_path.write_bytes(content)

    suffix = stored_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return add_uploaded_image_document(safe_name, stored_path, metadata)

    if suffix in PDF_SUFFIXES:
        return add_uploaded_pdf_document(safe_name, stored_path, metadata)

    if suffix in SPREADSHEET_SUFFIXES:
        return add_uploaded_spreadsheet_document(safe_name, stored_path, metadata)

    if suffix in HTML_SUFFIXES:
        return add_uploaded_html_document(safe_name, stored_path, metadata)

    if suffix in CAD_SUFFIXES:
        return add_uploaded_cad_document(safe_name, stored_path, metadata)

    if suffix not in TEXT_SUFFIXES:
        return {
            "filename": safe_name,
            "stored_path": str(stored_path),
            "added_chunks": 0,
            "warning": "Only text-like, PDF, spreadsheet, HTML, CAD text, and image files are parsed by the lightweight RAG uploader.",
        }

    text = content.decode("utf-8", errors="replace")
    documents = normalize_upload_document(safe_name, text, metadata)
    with USER_DOCUMENT_PATH.open("a", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    append_text_vectors(documents)

    return {
        "filename": safe_name,
        "stored_path": str(stored_path),
        "added_chunks": len(documents),
        "document_ids": [document["document_id"] for document in documents],
    }


def extract_pdf_text(path: Path) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        return {
            "text": "",
            "mode": "unavailable",
            "page_count": 0,
            "warning": f"PDF parser unavailable: {type(exc).__name__}: {exc}",
        }

    try:
        parts = []
        with fitz.open(path) as document:
            for index, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    parts.append(f"[Page {index}]\n{text}")
            page_count = document.page_count
        return {
            "text": "\n\n".join(parts).strip(),
            "mode": "pymupdf",
            "page_count": page_count,
            "warning": "" if parts else "No extractable PDF text was found.",
        }
    except Exception as exc:
        return {
            "text": "",
            "mode": "failed",
            "page_count": 0,
            "warning": f"PDF parsing failed: {type(exc).__name__}: {exc}",
        }


def add_uploaded_pdf_document(filename: str, stored_path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    parsed = extract_pdf_text(stored_path)
    if not parsed["text"]:
        return {
            "filename": filename,
            "stored_path": str(stored_path),
            "added_chunks": 0,
            "pdf": parsed,
            "warning": parsed["warning"],
        }

    pdf_metadata = {
        **metadata,
        "document_type": metadata.get("document_type") or "uploaded_pdf",
        "tags": metadata.get("tags") or ["uploaded", "pdf", "maintenance"],
    }
    documents = normalize_upload_document(filename, parsed["text"], pdf_metadata)
    for document in documents:
        document["pdf"] = {
            "mode": parsed["mode"],
            "page_count": parsed["page_count"],
            "warning": parsed["warning"],
        }

    with USER_DOCUMENT_PATH.open("a", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    append_text_vectors(documents)

    return {
        "filename": filename,
        "stored_path": str(stored_path),
        "added_chunks": len(documents),
        "document_ids": [document["document_id"] for document in documents],
        "pdf": {
            "mode": parsed["mode"],
            "page_count": parsed["page_count"],
            "warning": parsed["warning"],
        },
    }


def add_uploaded_spreadsheet_document(filename: str, stored_path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    try:
        import pandas as pd  # type: ignore

        sheets = pd.read_excel(stored_path, sheet_name=None)
        parts = []
        for sheet_name, frame in sheets.items():
            parts.append(f"[Sheet {sheet_name}]")
            parts.append(frame.fillna("").astype(str).to_csv(index=False))
        text = "\n".join(parts)
        work_order_records = infer_work_order_records(filename, sheets, metadata)
        mode = "pandas_openpyxl"
        warning = ""
    except Exception as exc:
        text = ""
        work_order_records = []
        mode = "failed"
        warning = f"Spreadsheet parsing failed: {type(exc).__name__}: {exc}"

    if not text.strip():
        return {"filename": filename, "stored_path": str(stored_path), "added_chunks": 0, "spreadsheet": {"mode": mode, "warning": warning}, "warning": warning}

    sheet_metadata = {
        **metadata,
        "document_type": metadata.get("document_type") or "uploaded_spreadsheet",
        "tags": metadata.get("tags") or ["uploaded", "spreadsheet", "maintenance"],
    }
    documents = normalize_upload_document(filename, text, sheet_metadata)
    for document in documents:
        document["spreadsheet"] = {"mode": mode, "warning": warning}
    with USER_DOCUMENT_PATH.open("a", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    append_text_vectors(documents)
    append_work_orders(work_order_records)
    return {
        "filename": filename,
        "stored_path": str(stored_path),
        "added_chunks": len(documents),
        "document_ids": [document["document_id"] for document in documents],
        "work_order_records": len(work_order_records),
        "spreadsheet": {"mode": mode, "warning": warning},
    }


def add_uploaded_html_document(filename: str, stored_path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    raw = stored_path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else metadata.get("title")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        text = soup.get_text("\n", strip=True)
        mode = "beautifulsoup"
    except Exception:
        title = metadata.get("title")
        text = re.sub(r"<[^>]+>", " ", raw)
        mode = "regex_html_strip"

    html_metadata = {
        **metadata,
        "title": title or metadata.get("title") or Path(filename).stem,
        "document_type": metadata.get("document_type") or "uploaded_html",
        "tags": metadata.get("tags") or ["uploaded", "html", "maintenance"],
    }
    documents = normalize_upload_document(filename, text, html_metadata)
    for document in documents:
        document["html"] = {"mode": mode}
    with USER_DOCUMENT_PATH.open("a", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    append_text_vectors(documents)
    return {"filename": filename, "stored_path": str(stored_path), "added_chunks": len(documents), "document_ids": [document["document_id"] for document in documents], "html": {"mode": mode}}


def add_uploaded_cad_document(filename: str, stored_path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    raw = stored_path.read_text(encoding="utf-8", errors="replace")
    suffix = stored_path.suffix.lower()
    if suffix == ".svg":
        text = re.sub(r"<[^>]+>", " ", raw)
        mode = "svg_text_metadata"
    else:
        text = raw
        mode = "dxf_text_metadata"
    cad_metadata = {
        **metadata,
        "document_type": metadata.get("document_type") or "uploaded_cad_drawing",
        "tags": metadata.get("tags") or ["uploaded", "cad", "drawing", "maintenance"],
    }
    documents = normalize_upload_document(filename, text, cad_metadata)
    for document in documents:
        document["cad"] = {"mode": mode, "warning": "CAD support extracts searchable drawing text/metadata, not geometric reasoning."}
    with USER_DOCUMENT_PATH.open("a", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    append_text_vectors(documents)
    return {"filename": filename, "stored_path": str(stored_path), "added_chunks": len(documents), "document_ids": [document["document_id"] for document in documents], "cad": {"mode": mode}}


def ocr_image(path: Path) -> dict[str, Any]:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except Exception as exc:
        return {
            "text": "",
            "mode": "unavailable",
            "warning": f"OCR unavailable: {type(exc).__name__}: {exc}",
        }

    try:
        text = pytesseract.image_to_string(Image.open(path))
        return {
            "text": text.strip(),
            "mode": "pytesseract",
            "warning": "",
        }
    except Exception as exc:
        return {
            "text": "",
            "mode": "failed",
            "warning": f"OCR failed: {type(exc).__name__}: {exc}",
        }


def dinov2_image_summary(path: Path) -> dict[str, Any]:
    try:
        from vision_runtime import dinov2_global_embedding, infer_fault_from_path, score_image

        output_path = UPLOAD_ROOT / f"{path.stem}_dinov2_rag.png"
        predicted_fault = infer_fault_from_path(str(path), "visual_anomaly")
        metrics = score_image(path, output_path, predicted_fault)
        embedding = dinov2_global_embedding(path)
        return {
            "mode": "dinov2_patch_memory",
            "predicted_fault": predicted_fault,
            "anomaly_score": metrics.get("anomaly_score"),
            "is_anomaly": metrics.get("is_anomaly"),
            "image_threshold": metrics.get("threshold"),
            "pixel_threshold": metrics.get("pixel_threshold"),
            "result_image": str(output_path),
            "embedding": embedding,
            "warning": "",
        }
    except Exception as exc:
        return {
            "mode": "failed",
            "predicted_fault": "visual_anomaly",
            "anomaly_score": None,
            "is_anomaly": None,
            "image_threshold": None,
            "pixel_threshold": None,
            "result_image": None,
            "embedding": None,
            "warning": f"DINOv2 image indexing failed: {type(exc).__name__}: {exc}",
        }


def add_uploaded_image_document(filename: str, stored_path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    ocr = ocr_image(stored_path)
    vision = dinov2_image_summary(stored_path)
    layout = analyze_image_layout(stored_path)
    image_text = "\n".join(
        part
        for part in [
            f"Uploaded SOP image file: {filename}",
            f"Title: {metadata.get('title') or stored_path.stem}",
            f"Asset type: {metadata.get('asset_type') or 'industrial_asset'}",
            f"Tags: {metadata.get('tags') or 'uploaded, sop, maintenance, image'}",
            f"Layout analysis: {layout['layout_type']} with confidence {layout['confidence']}.",
            f"OCR text: {ocr['text']}" if ocr["text"] else "",
            (
                "DINOv2 visual summary: "
                f"predicted_fault={vision['predicted_fault']}; "
                f"is_anomaly={vision['is_anomaly']}; "
                f"anomaly_score={vision['anomaly_score']}; "
                f"image_threshold={vision['image_threshold']}; "
                f"pixel_threshold={vision['pixel_threshold']}."
            ),
            ocr["warning"],
            vision["warning"],
        ]
        if part
    )
    image_metadata = {
        **metadata,
        "document_type": metadata.get("document_type") or "uploaded_sop_image",
        "tags": metadata.get("tags") or ["uploaded", "sop", "maintenance", "image", vision["predicted_fault"]],
    }
    documents = normalize_upload_document(filename, image_text, image_metadata)
    for document in documents:
        document["ocr"] = {"mode": ocr["mode"], "warning": ocr["warning"]}
        document["vision"] = {key: value for key, value in vision.items() if key != "embedding"}
        document["layout"] = layout

    with USER_DOCUMENT_PATH.open("a", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    append_text_vectors(documents)

    if vision.get("embedding"):
        for document in documents:
            append_visual_index(
                {
                    "document_id": document["document_id"],
                    "source_file": filename,
                    "stored_path": str(stored_path),
                    "layout_type": layout["layout_type"],
                    "predicted_fault": vision.get("predicted_fault"),
                    "embedding": vision["embedding"],
                }
            )

    return {
        "filename": filename,
        "stored_path": str(stored_path),
        "added_chunks": len(documents),
        "document_ids": [document["document_id"] for document in documents],
        "ocr": {"mode": ocr["mode"], "warning": ocr["warning"]},
        "layout": layout,
        "vision": {key: value for key, value in vision.items() if key != "embedding"},
    }


def collection_summary() -> dict[str, Any]:
    documents = load_documents()
    uploaded = [document for document in documents if document.get("source_status") == "uploaded"]
    visual_records = load_visual_index()
    text_records = load_text_vector_index()
    work_order_records = load_work_orders()
    if len(text_records) < len(documents):
        rebuild_text_vector_index(documents)
        text_records = load_text_vector_index()
    return {
        "document_count": len(documents),
        "uploaded_chunk_count": len(uploaded),
        "visual_embedding_count": len(visual_records),
        "text_embedding_count": len(text_records),
        "work_order_record_count": len(work_order_records),
        "base_document_path": str(DOCUMENT_PATH),
        "uploaded_document_path": str(USER_DOCUMENT_PATH),
        "visual_index_path": str(VISUAL_INDEX_PATH),
        "text_vector_index_path": str(TEXT_VECTOR_INDEX_PATH),
        "work_order_index_path": str(WORK_ORDER_INDEX_PATH),
        "mode": "local_vector_tfidf_keyword_visual_hybrid",
        "supports": {
            "text_files": sorted(TEXT_SUFFIXES),
            "pdf_files": sorted(PDF_SUFFIXES),
            "spreadsheet_files": sorted(SPREADSHEET_SUFFIXES),
            "html_files": sorted(HTML_SUFFIXES),
            "cad_text_files": sorted(CAD_SUFFIXES),
            "image_files": sorted(IMAGE_SUFFIXES),
            "pdf_reader": "pymupdf_fitz",
            "spreadsheet_reader": "pandas_openpyxl",
            "work_order_search": "row_level_spare_parts_damage_search",
            "html_reader": "beautifulsoup",
            "image_ocr": "optional_pytesseract",
            "text_embedding": f"local_hashing_{LOCAL_VECTOR_FEATURES}",
            "vector_db": vector_backend_status(),
            "image_visual_indexing": "dinov2_patch_memory",
            "visual_matching": "local_numpy_cosine",
            "layout_analysis": "cv_regions_plus_dinov2_metadata",
        },
    }


def vector_backend_status() -> dict[str, Any]:
    pinecone_ready = bool(os.environ.get("PINECONE_API_KEY") and os.environ.get("PINECONE_INDEX"))
    try:
        import pinecone  # type: ignore  # noqa: F401

        pinecone_installed = True
    except Exception:
        pinecone_installed = False
    try:
        import chromadb  # type: ignore  # noqa: F401

        chroma_installed = True
    except Exception:
        chroma_installed = False
    selected = "pinecone" if pinecone_ready and pinecone_installed else "chromadb" if chroma_installed else "local_jsonl_hashing"
    return {
        "selected": selected,
        "pinecone_installed": pinecone_installed,
        "pinecone_configured": pinecone_ready,
        "chromadb_installed": chroma_installed,
        "fallback": "local_jsonl_hashing",
    }


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 1
    }


def document_text(document: dict[str, Any]) -> str:
    return " ".join(
        [
            str(document.get("document_id", "")),
            str(document.get("title", "")),
            str(document.get("asset_type", "")),
            str(document.get("section", "")),
            " ".join(str(tag) for tag in document.get("tags", [])),
            str(document.get("text", "")),
        ]
    )


def extract_codes(text: str) -> set[str]:
    return {match.upper() for match in re.findall(r"\b[A-Z]{1,5}-?[A-Z]?\d{2,4}\b", text.upper())}


def keyword_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    overlap = query_tokens & doc_tokens
    return len(overlap) / math.sqrt(max(len(query_tokens) * len(doc_tokens), 1))


def tfidf_scores(query: str, documents: list[dict[str, Any]]) -> list[float] | None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None

    corpus = [document_text(document) for document in documents]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    return cosine_similarity(query_vector, matrix)[0].tolist()


def local_vector_scores(query: str, documents: list[dict[str, Any]]) -> dict[str, float]:
    records = load_text_vector_index()
    if len(records) < len(documents):
        rebuild_text_vector_index(documents)
        records = load_text_vector_index()
    query_embedding = local_text_embedding(query)
    scores = {}
    for record in records:
        document_id = record.get("document_id")
        embedding = record.get("embedding")
        if document_id and isinstance(embedding, list):
            scores[str(document_id)] = cosine_similarity_list(query_embedding, embedding)
    return scores


def cosine_similarity_list(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def visual_scores_for_query(payload: dict[str, Any]) -> dict[str, float]:
    image_path = payload.get("image_path")
    if not isinstance(image_path, str) or not Path(image_path).is_file():
        return {}
    visual_records = load_visual_index()
    if not visual_records:
        return {}
    try:
        from vision_runtime import dinov2_global_embedding

        query_embedding = dinov2_global_embedding(Path(image_path))
    except Exception:
        return {}

    scores: dict[str, float] = {}
    for record in visual_records:
        embedding = record.get("embedding")
        document_id = record.get("document_id")
        if isinstance(embedding, list) and document_id:
            scores[str(document_id)] = cosine_similarity_list(query_embedding, embedding)
    return scores


def build_query(
    payload: dict[str, Any],
    telemetry: dict[str, Any],
    vision: dict[str, Any],
) -> str:
    parts = [
        str(payload.get("description", "")),
        str(payload.get("asset_type", "")),
        str(payload.get("location", "")),
        str(payload.get("operating_state", "")),
        f"telemetry severity {telemetry.get('severity')}",
        f"predicted RUL {telemetry.get('predicted_rul')}",
        f"failure risk {telemetry.get('failure_risk')}",
        f"vision anomaly {vision.get('is_anomaly')}",
        f"defect {vision.get('predicted_fault')}",
        f"vision severity {vision.get('severity')}",
    ]

    for key in ("telemetry_assessment", "vision_assessment", "maintenance_recommendation"):
        value = payload.get(key)
        if isinstance(value, dict):
            parts.append(json.dumps(value))
    hypotheses = payload.get("root_cause_hypothesis")
    if isinstance(hypotheses, list):
        parts.extend(str(item) for item in hypotheses)
    return " ".join(part for part in parts if part and part != "None")


def retrieve(
    payload: dict[str, Any],
    telemetry: dict[str, Any],
    vision: dict[str, Any],
    top_k: int = TOP_K,
) -> dict[str, Any]:
    started = time.perf_counter()
    documents = load_documents()
    query = build_query(payload, telemetry, vision)
    if not documents:
        return {
            "query": query,
            "results": [],
            "latency_ms": 0.0,
            "mode": "missing_knowledge_base",
        }

    query_tokens = tokenize(query)
    query_codes = extract_codes(query)
    semantic = tfidf_scores(query, documents)
    vector_scores = local_vector_scores(query, documents)
    visual_scores = visual_scores_for_query(payload)

    scored: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        doc_text = document_text(document)
        doc_tokens = tokenize(doc_text)
        kw_score = keyword_score(query_tokens, doc_tokens)
        semantic_score = semantic[index] if semantic is not None else kw_score
        vector_score = max(0.0, vector_scores.get(str(document.get("document_id")), 0.0))
        document_id = str(document.get("document_id", "")).upper()
        code_boost = 1.0 if any(code in document_id for code in query_codes) else 0.0
        tag_boost = 0.15 if str(vision.get("predicted_fault", "")).lower() in " ".join(document.get("tags", [])).lower() else 0.0
        visual_score = max(0.0, visual_scores.get(str(document.get("document_id")), 0.0))
        score = 0.35 * semantic_score + 0.25 * vector_score + 0.20 * kw_score + 0.20 * visual_score + 0.2 * code_boost + tag_boost
        scored.append(
            {
                "document": document,
                "semantic_score": float(semantic_score),
                "vector_score": float(vector_score),
                "keyword_score": float(kw_score),
                "visual_score": float(visual_score),
                "score": float(score),
            }
        )

    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]
    results = []
    for item in ranked:
        document = item["document"]
        results.append(
            {
                "document_id": document.get("document_id"),
                "title": document.get("title"),
                "document_type": document.get("document_type"),
                "asset_type": document.get("asset_type"),
                "revision": document.get("revision"),
                "section": document.get("section"),
                "text": document.get("text"),
                "tags": document.get("tags", []),
                "source_status": document.get("source_status"),
                "semantic_score": item["semantic_score"],
                "vector_score": item["vector_score"],
                "keyword_score": item["keyword_score"],
                "visual_score": item["visual_score"],
                "score": item["score"],
            }
        )

    return {
        "query": query,
        "results": results,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "mode": "local_vector_tfidf_keyword_visual_hybrid",
        "visual_match_count": len(visual_scores),
        "text_vector_match_count": len(vector_scores),
        "vector_backend": vector_backend_status(),
    }
