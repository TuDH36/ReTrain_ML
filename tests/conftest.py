from dataclasses import replace

import pytest

from retrain_app.config import Settings
from retrain_app.contracts import StartRequest
from retrain_app.store import Store


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path / "state", api_token="test-token-abcdefghijklmnopqrstuvwxyz",
                    callback_url="http://be.test/jobs/{job_id}/status", callback_token="callback-secret",
                    es_url="http://es.test", minio_endpoint="http://s3.test",
                    minio_access_key="test", minio_secret_key="test", es_mapping_confirmed=True,
                    train_threads=1)


@pytest.fixture
def request_body():
    return StartRequest(job_id="RT-1", tenant_id="VND", base_version="v1.0", target_version="v1.1",
                        added_alert_ids=["NewID_A", "NewID_B"], removed_alert_ids=[]).model_dump()


@pytest.fixture
def store(settings):
    return Store(settings.db_path)
