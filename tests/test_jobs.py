import asyncio
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from retrain_app.api import create_app
from retrain_app.contracts import Cancelled, JobError
from retrain_app.store import Store
from retrain_app.supervisor import Callbacks, InstanceLock, Supervisor


def test_api_accept_replay_tenant_isolation_and_cancel(settings, request_body):
    with TestClient(create_app(settings, run_supervisor=False)) as client:
        path = "/api/v1/ai/retrain/start"
        assert client.post(path, json=request_body).status_code == 401
        headers = {"Authorization": f"Bearer {settings.api_token}"}
        bad = dict(request_body, target_version="../../other")
        assert client.post(path, json=bad, headers=headers).status_code == 422
        first = client.post(path, json=request_body, headers=headers)
        assert first.status_code == 202 and first.json()["status"] == "QUEUED"
        assert client.post(path, json=request_body, headers=headers).json() == first.json()
        assert client.post(path, json=dict(request_body, added_alert_ids=[]), headers=headers).status_code == 409
        assert client.post(path, json=dict(request_body, job_id="RT-2", target_version="v1.2"), headers=headers).status_code == 409
        assert client.post(path, json=dict(request_body, job_id="RT-3", tenant_id="BIDV"), headers=headers).status_code == 202
        cancel = "/api/v1/ai/retrain/cancel"
        assert client.post(cancel, json={"job_id": "RT-1", "tenant_id": "BIDV"}, headers=headers).status_code == 404
        result = client.post(cancel, json={"job_id": "RT-1", "tenant_id": "VND"}, headers=headers)
        assert result.status_code == 200 and result.json()["status"] == "CANCELLED"
        assert client.post(path, json=request_body, headers=headers).json()["status"] == "CANCELLED"
        assert client.post(path, json=dict(request_body, job_id="RT-4", target_version="v1.2"), headers=headers).status_code == 202


def test_mapping_requires_explicit_confirmation(settings, request_body):
    with TestClient(create_app(replace(settings, es_mapping_confirmed=False), run_supervisor=False)) as client:
        assert client.post("/api/v1/ai/retrain/start", json=request_body,
                           headers={"Authorization": f"Bearer {settings.api_token}"}).status_code == 503


def test_concurrent_starts_one_winner(store, request_body):
    def submit(i):
        try:
            store.accept(dict(request_body, job_id=f"RT-{i}", target_version=f"v1.{i}"), 50)
            return True
        except JobError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(submit, range(16))) == 1


def test_restart_and_publish_cancellation_are_durable(settings, store, request_body):
    store.accept(request_body, 50)
    store.transition("RT-1", {"QUEUED"}, "TRAIN")
    reopened = Store(settings.db_path)
    reopened.recover()
    assert reopened.get("RT-1")["status"] == "FAILED"
    other = dict(request_body, job_id="RT-2", target_version="v1.2")
    store.accept(other, 50)
    for old, new in [("QUEUED", "TRAIN"), ("TRAIN", "EVALUATE"), ("EVALUATE", "PUBLISH")]:
        store.transition("RT-2", {old}, new)
    reopened.recover()
    assert reopened.get("RT-2")["status"] == "PUBLISH"
    store.request_cancel("RT-2", "VND")
    with pytest.raises(Cancelled):
        store.transition("RT-2", {"PUBLISH"}, "DONE")
    reopened.recover()
    assert reopened.get("RT-2")["status"] == "CANCELLING"
    with pytest.raises(JobError):
        store.accept(dict(other, job_id="RT-3", target_version="v1.3"), 50)
    store.cancel_cleaned("RT-2")
    assert reopened.get("RT-2")["status"] == "CANCELLED"


def test_done_cancel_race_has_one_terminal_outcome(store, request_body):
    for i in range(20):
        job = dict(request_body, job_id=f"RACE-{i}", tenant_id=f"TENANT{i}")
        store.accept(job, 50)
        for old, new in [("QUEUED", "TRAIN"), ("TRAIN", "EVALUATE"), ("EVALUATE", "PUBLISH")]:
            store.transition(job["job_id"], {old}, new)
        def finish():
            try:
                store.transition(job["job_id"], {"PUBLISH"}, "DONE")
            except Cancelled:
                pass
        def cancel():
            try:
                store.request_cancel(job["job_id"], job["tenant_id"])
            except JobError as exc:
                assert exc.code == "JOB_NOT_CANCELLABLE"
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda f: f(), [finish, cancel]))
        assert store.get(job["job_id"])["status"] in {"DONE", "CANCELLING"}


def test_callback_retries_preserve_body_sequence_and_survive_restart(settings, store, request_body):
    store.accept(request_body, 50)
    store.transition("RT-1", {"QUEUED"}, "TRAIN")
    store.transition("RT-1", {"TRAIN"}, "EVALUATE")
    received = []
    def handler(request):
        received.append((json.loads(request.content), request.headers["X-Event-Seq"]))
        assert request.headers["Authorization"] == "Bearer callback-secret"
        return httpx.Response(503 if len(received) == 1 else 204)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cb = Callbacks(settings, store)
            event = store.due_callbacks()[0]
            await cb.deliver(client, event)
            assert store.pending_callback_count() == 2
            # Clock-independent retry: same persisted event, same sequence.
            cb = Callbacks(settings, Store(settings.db_path))
            await cb.deliver(client, event)
            next_event = store.due_callbacks()[0]
            assert next_event["seq"] == 2
            await cb.deliver(client, next_event)
    asyncio.run(scenario())
    assert received == [({"status": "TRAIN"}, "1"), ({"status": "TRAIN"}, "1"), ({"status": "EVALUATE"}, "2")]
    assert store.pending_callback_count() == 0


def test_cancel_stops_only_its_process_and_waits_for_cleanup(settings, store, request_body, monkeypatch):
    supervisor = Supervisor(settings, store)
    monkeypatch.setattr(supervisor, "spawn", lambda job: None)
    processes = []
    try:
        for job_id, tenant in [("RT-1", "VND"), ("RT-2", "BIDV")]:
            store.accept(dict(request_body, job_id=job_id, tenant_id=tenant), 50)
            store.transition(job_id, {"QUEUED"}, "TRAIN")
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=os.name != "nt")
            processes.append(proc)
            supervisor.children[job_id] = {"process": proc, "started": time.monotonic(), "term_at": None, "cleanup": False}
        store.request_cancel("RT-1", "VND")
        supervisor.tick()
        processes[0].wait(timeout=10)
        supervisor.tick()
        assert processes[1].poll() is None
        assert store.get("RT-1")["status"] == "CANCELLING"
        # A successfully exited cleanup process is required before final acknowledgement.
        cleaner = subprocess.Popen([sys.executable, "-c", "pass"])
        cleaner.wait(timeout=10)
        supervisor.children["RT-1"] = {"process": cleaner, "started": time.monotonic(), "term_at": None, "cleanup": True}
        supervisor.tick()
        assert store.get("RT-1")["status"] == "CANCELLED"
        assert store.get("RT-2")["status"] == "TRAIN"
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def test_second_supervisor_rejected(settings):
    settings.state_dir.mkdir()
    lock = InstanceLock(settings.state_dir / "service.lock")
    try:
        with pytest.raises(RuntimeError):
            InstanceLock(settings.state_dir / "service.lock")
    finally:
        lock.close()


def test_cancel_before_publish_does_not_require_minio(settings, store, request_body, monkeypatch):
    from retrain_app.config import Settings
    from retrain_app.runner import cleanup
    import retrain_app.storage as storage

    store.accept(request_body, 50)
    store.transition("RT-1", {"QUEUED"}, "TRAIN")
    store.request_cancel("RT-1", "VND")
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    def forbidden(*args):
        raise AssertionError("MinIO must not be contacted for a TRAIN cancellation")
    monkeypatch.setattr(storage, "Models", forbidden)
    cleanup("RT-1")
    assert store.get("RT-1")["status"] == "CANCELLING"
    store.cancel_cleaned("RT-1")
    assert store.get("RT-1")["status"] == "CANCELLED"
