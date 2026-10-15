"""Real API supervisor, subprocess ML and HTTP clients, with isolated fake ES/S3/BE."""
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

import pytest
from fastapi.testclient import TestClient

from retrain_app.api import create_app
from retrain_app.storage import ARTIFACTS, version_prefix
from test_training import synthetic_base, synthetic_rows


class Services:
    def __init__(self, tmp_path):
        directory = tmp_path / "base"
        manifest = synthetic_base(directory)
        rows = synthetic_rows()
        manifest["training_alert_ids"] = [r["alert_id"] for r in rows[:30]]
        prefix = "/soar-models/" + version_prefix("VND", "v1.0")
        self.objects = {prefix + name: ((directory / name).read_bytes(), {}) for name in ARTIFACTS}
        self.objects[prefix + "manifest.json"] = (json.dumps(manifest).encode(), {})
        self.rows = {r["alert_id"]: r for r in rows}
        self.callbacks = []
        self.block_upload = False
        self.upload_started = threading.Event()
        self.release_upload = threading.Event()
        self.fail_callback_once = True
        self.fail_delete = False
        self.delete_attempted = threading.Event()
        services = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, status, body=b"", headers=None):
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def body(self):
                return self.rfile.read(int(self.headers.get("Content-Length", 0)))

            def do_GET(self):
                key = unquote(urlsplit(self.path).path)
                if key not in services.objects:
                    self.reply(404, b"<Error><Code>NoSuchKey</Code></Error>")
                else:
                    data, metadata = services.objects[key]
                    self.reply(200, data, {"x-amz-meta-" + k: v for k, v in metadata.items()})

            do_HEAD = do_GET

            def do_PUT(self):
                key = unquote(urlsplit(self.path).path)
                data = self.body()
                if self.headers.get("If-None-Match") != "*":
                    self.reply(400)
                    return
                if key in services.objects:
                    self.reply(412, b"<Error><Code>PreconditionFailed</Code></Error>")
                    return
                if services.block_upload and key.endswith("cat_encoder.pkl"):
                    services.upload_started.set()
                    services.release_upload.wait(timeout=30)
                metadata = {k.lower()[11:]: v for k, v in self.headers.items() if k.lower().startswith("x-amz-meta-")}
                services.objects[key] = (data, metadata)
                self.reply(200, headers={"ETag": '"test-etag"'})

            def do_DELETE(self):
                if self.path == "/_search/scroll":
                    self.body()
                    self.reply(200, {"succeeded": True})
                elif services.fail_delete:
                    services.delete_attempted.set()
                    self.reply(503, b"<Error><Code>ServiceUnavailable</Code></Error>")
                else:
                    services.objects.pop(unquote(self.path), None)
                    self.reply(204)

            def do_POST(self):
                data = json.loads(self.body())
                if self.path.endswith("/_mget"):
                    docs = []
                    for item in data["docs"]:
                        row = services.rows[item["_id"]]
                        docs.append({"_id": item["_id"], "found": True, "_source": {
                            "id": item["_id"], "customerCode": "VND",
                            "rawData": {"ai_training": {"label": "TRUE_POSITIVE" if row["label"] else "FALSE_POSITIVE"}}}})
                    self.reply(200, {"docs": docs})
                elif "/_search?" in self.path:
                    refs = next(iter(data["query"]["terms"].values()))
                    hits = [{"_index": "case_events", "_id": "event-" + ref,
                             "_source": {"alertId": ref, "event": json.dumps(services.rows[ref]["events"])}} for ref in refs]
                    self.reply(200, {"hits": {"total": {"value": len(hits), "relation": "eq"}, "hits": hits}})
                elif self.path.startswith("/jobs/"):
                    services.callbacks.append((self.path, self.headers.get("X-Event-Seq"), data))
                    failed = services.fail_callback_once
                    services.fail_callback_once = False
                    self.reply(503 if failed else 204)
                else:
                    self.reply(404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.release_upload.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def wait_for(check, seconds=45):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.1)
    raise AssertionError("Timed out waiting for integration condition")


@pytest.fixture
def services(tmp_path):
    svc = Services(tmp_path)
    try:
        yield svc
    finally:
        svc.close()


def test_real_subprocess_end_to_end_and_callback_retry(settings, services):
    settings = replace(settings, es_url=services.url, minio_endpoint=services.url,
                       callback_url=services.url + "/jobs/{job_id}/status")
    app = create_app(settings)
    request = {"job_id": "HTTP-1", "tenant_id": "VND", "base_version": "v1.0", "target_version": "v1.1",
               "added_alert_ids": [f"CaseSensitiveID-{i}" for i in range(30, 40)], "removed_alert_ids": []}
    headers = {"Authorization": f"Bearer {settings.api_token}"}
    with TestClient(app) as client:
        response = client.post("/api/v1/ai/retrain/start", json=request, headers=headers)
        assert response.status_code == 202
        store = app.state.store
        wait_for(lambda: store.get("HTTP-1")["status"] in {"DONE", "FAILED"})
        assert store.get("HTTP-1")["status"] == "DONE", store.get("HTTP-1")["error_code"]
        wait_for(lambda: store.pending_callback_count() == 0)
        received = [(seq, body["status"]) for path, seq, body in services.callbacks]
        assert received == [("1", "TRAIN"), ("1", "TRAIN"), ("2", "EVALUATE"), ("3", "PUBLISH"), ("4", "DONE")]
        manifest_key = "/soar-models/" + response.json()["manifest_key"]
        manifest = json.loads(services.objects[manifest_key][0])
        assert len(manifest["training_alert_ids"]) == 40
        assert manifest["evaluation"]["dataset"] == "train"
        assert client.post("/api/v1/ai/retrain/start", json=request, headers=headers).status_code == 200


def test_cancel_publish_waits_for_inflight_put_then_cleans(settings, services):
    services.block_upload = True
    settings = replace(settings, es_url=services.url, minio_endpoint=services.url,
                       callback_url=services.url + "/jobs/{job_id}/status")
    app = create_app(settings)
    request = {"job_id": "HTTP-CANCEL", "tenant_id": "VND", "base_version": "v1.0", "target_version": "v1.2"}
    headers = {"Authorization": f"Bearer {settings.api_token}"}
    with TestClient(app) as client:
        assert client.post("/api/v1/ai/retrain/start", json=request, headers=headers).status_code == 202
        wait_for(services.upload_started.is_set)
        result = client.post("/api/v1/ai/retrain/cancel", json={"job_id": "HTTP-CANCEL", "tenant_id": "VND"}, headers=headers)
        assert result.status_code == 202 and result.json()["status"] == "CANCELLING"
        # Publish request is still running; CANCELLED must not be emitted yet.
        time.sleep(0.5)
        assert app.state.store.get("HTTP-CANCEL")["status"] == "CANCELLING"
        busy = client.post("/api/v1/ai/retrain/start", json=dict(request, job_id="HTTP-OTHER", target_version="v1.3"), headers=headers)
        assert busy.status_code == 409
        services.fail_delete = True
        services.release_upload.set()
        wait_for(services.delete_attempted.is_set)
        assert app.state.store.get("HTTP-CANCEL")["status"] == "CANCELLING"
        services.fail_delete = False
        wait_for(lambda: app.state.store.get("HTTP-CANCEL")["status"] == "CANCELLED")
        wait_for(lambda: app.state.store.pending_callback_count() == 0)
        assert not any("/versions/v1.2/" in k for k in services.objects)
        assert len([k for k in services.objects if "/versions/v1.0/" in k]) == 5
        statuses = [body["status"] for _, _, body in services.callbacks]
        assert statuses[-1] == "CANCELLED" and "DONE" not in statuses
