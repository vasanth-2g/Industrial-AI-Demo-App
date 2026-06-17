from __future__ import annotations

import cgi
import json
import mimetypes
import os
import re
import shutil
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from rag_runtime import add_uploaded_document, collection_summary, search_work_orders
from scenario_runtime import run_scenario


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
FRONTEND_ROOT = REPO_ROOT / "app" / "frontend"
RUNTIME_ROOT = APP_ROOT / "runtime"
RUNS_ROOT = RUNTIME_ROOT / "runs"


def default_demo_package() -> Path:
    return APP_ROOT.resolve()


APP_PACKAGE = default_demo_package()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def safe_upload_name(filename: str) -> str:
    name = Path(filename).name.strip() or "uploaded_image"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:120] or "uploaded_image"


def reset_runtime() -> None:
    runtime = RUNTIME_ROOT.resolve()
    app_root = APP_ROOT.resolve()
    if runtime.exists():
        if app_root not in runtime.parents:
            raise RuntimeError(f"Refusing to clear runtime outside app: {runtime}")
        shutil.rmtree(runtime)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def make_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = RUNS_ROOT / stamp
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_demo_payload() -> dict:
    return {
        "app_root": str(APP_ROOT),
        "runtime_root": str(RUNTIME_ROOT),
        "health": {
            "telemetry_models": str(APP_ROOT / "backend" / "XGboost" / "telemetry_risk"),
            "vision_models": str(APP_ROOT / "backend" / "vision_dinov2"),
            "rag_documents": str(APP_ROOT / "backend" / "knowledge_rag"),
        },
    }


class DemoHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, body: str, content_type: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_json({"error": "file not found", "path": str(path)}, 404)
            return

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = (
            self.rfile.read(content_length).decode("utf-8")
            if content_length
            else "{}"
        )
        if not raw_body.strip():
            return {}
        return json.loads(raw_body)

    def read_infer_body(self, run_dir: Path) -> dict:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            return self.read_json_body()

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        payload: dict = {}
        uploaded_image: dict = {}

        scenario_field = form["scenario"] if "scenario" in form else None
        if scenario_field is None and "payload" in form:
            scenario_field = form["payload"]
        if scenario_field is not None and not isinstance(scenario_field, list):
            value = scenario_field.value.strip()
            if value:
                payload.update(json.loads(value))

        for key in form.keys():
            field = form[key]
            fields = field if isinstance(field, list) else [field]
            for item in fields:
                if item.filename:
                    filename = safe_upload_name(item.filename)
                    output_path = run_dir / "inputs" / filename
                    suffix = output_path.suffix
                    stem = output_path.stem
                    counter = 1
                    while output_path.exists():
                        output_path = run_dir / "inputs" / f"{stem}_{counter}{suffix}"
                        counter += 1
                    data = item.file.read()
                    output_path.write_bytes(data)
                    if key in {"input", "scenario_file", "scenario"} or filename.lower().endswith(".txt"):
                        text = data.decode("utf-8-sig", errors="replace").strip()
                        parsed = False
                        if text:
                            try:
                                payload.update(json.loads(text))
                                parsed = True
                            except json.JSONDecodeError:
                                payload["description"] = text
                        payload["input_path"] = str(output_path)
                        payload["input_filename"] = filename
                        payload["input_bytes"] = len(data)
                        payload["input_parse_mode"] = "json" if parsed else "text"
                    else:
                        uploaded_image = {
                            "image_path": str(output_path),
                            "image_filename": filename,
                            "image_bytes": len(data),
                        }
                elif key not in {"scenario", "payload"}:
                    payload[key] = item.value

        if uploaded_image:
            payload.update(uploaded_image)

        return payload

    def read_multipart_form(self) -> cgi.FieldStorage:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            raise ValueError("Expected multipart/form-data")
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        try:
            if path == "/api/health":
                self.send_json(
                    {
                        "status": "ok",
                        "app_root": str(APP_ROOT),
                        "frontend_root": str(FRONTEND_ROOT),
                        "runtime_root": str(RUNTIME_ROOT),
                    }
                )
                return

            if path == "/api/demo":
                self.send_json(build_demo_payload())
                return

            if path == "/api/rag/documents":
                self.send_json(collection_summary())
                return

            if path == "/api/work-orders/search":
                query = unquote(parsed.query or "")
                params = {}
                for part in query.split("&"):
                    if "=" in part:
                        key, value = part.split("=", 1)
                        params[key] = value
                self.send_json(search_work_orders(params.get("q", "")))
                return

            if path.startswith("/api/runtime/"):
                relative = path.removeprefix("/api/runtime/")
                file_path = (RUNTIME_ROOT / relative).resolve()
                runtime_root = RUNTIME_ROOT.resolve()
                if runtime_root not in file_path.parents and file_path != runtime_root:
                    self.send_json({"error": "invalid runtime path"}, 403)
                    return
                self.send_file(file_path)
                return

            if path == "/":
                path = "/index.html"

            static_path = (FRONTEND_ROOT / path.lstrip("/")).resolve()
            if FRONTEND_ROOT not in static_path.parents and static_path != FRONTEND_ROOT:
                self.send_json({"error": "invalid static path"}, 403)
                return

            if static_path.is_file():
                self.send_file(static_path)
                return

            self.send_json({"error": "not found", "path": path}, 404)

        except FileNotFoundError as exc:
            self.send_json({"error": "missing required artifact", "path": str(exc)}, 500)
        except Exception as exc:  # defensive demo server fallback
            self.send_json({"error": type(exc).__name__, "message": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        try:
            if path == "/api/infer":
                run_dir = make_run_dir()
                payload = self.read_infer_body(run_dir)
                (run_dir / "inputs" / "scenario.json").write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )
                result = run_scenario(payload, APP_PACKAGE, run_dir)
                heatmap_path = result.get("vision", {}).get("heatmap_path")
                if heatmap_path:
                    heatmap_file = Path(heatmap_path).resolve()
                    if heatmap_file.is_file() and RUNTIME_ROOT.resolve() in heatmap_file.parents:
                        result["vision"]["heatmap_url"] = (
                            "/api/runtime/"
                            + heatmap_file.relative_to(RUNTIME_ROOT.resolve()).as_posix()
                        )
                result["run"] = {
                    "run_dir": str(run_dir),
                    "input_json": str(run_dir / "inputs" / "scenario.json"),
                    "output_json": str(run_dir / "outputs" / "result.json"),
                }
                (run_dir / "outputs" / "result.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )
                self.send_json(result)
                return

            if path == "/api/rag/documents":
                form = self.read_multipart_form()
                metadata = {}
                if "metadata" in form and not isinstance(form["metadata"], list):
                    raw_metadata = form["metadata"].value.strip()
                    if raw_metadata:
                        metadata = json.loads(raw_metadata)

                uploaded = []
                for key in form.keys():
                    field = form[key]
                    fields = field if isinstance(field, list) else [field]
                    for item in fields:
                        if item.filename:
                            uploaded.append(
                                add_uploaded_document(
                                    item.filename,
                                    item.file.read(),
                                    metadata,
                                )
                            )

                self.send_json(
                    {
                        "uploaded": uploaded,
                        "collection": collection_summary(),
                    }
                )
                return

            if path == "/api/work-orders/search":
                body = self.read_json_body()
                query = str(body.get("query") or body.get("q") or "")
                limit = int(body.get("limit") or 8)
                self.send_json(search_work_orders(query, limit=limit))
                return

            self.send_json({"error": "not found", "path": path}, 404)
        except json.JSONDecodeError as exc:
            self.send_json({"error": "invalid_json", "message": str(exc)}, 400)
        except Exception as exc:  # defensive demo server fallback
            self.send_json({"error": type(exc).__name__, "message": str(exc)}, 500)


def main() -> None:
    reset_runtime()
    host = os.environ.get("DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("DEMO_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"App root: {APP_ROOT}")
    print(f"Runtime root: {RUNTIME_ROOT}")
    print(f"Serving dashboard: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
