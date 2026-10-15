import io
import json

import pytest
from botocore.exceptions import ClientError

from retrain_app.contracts import Cancelled, JobError
from retrain_app.storage import ARTIFACTS, Models, file_info, version_prefix


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.writes = []
        self.on_put = lambda key: None

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data, metadata = self.objects[Key]
        return {"Body": io.BytesIO(data), "ContentLength": len(data), "Metadata": metadata}

    def head_object(self, **kwargs):
        result = self.get_object(**kwargs)
        result["Body"].close()
        return result

    def put_object(self, Bucket, Key, Body, Metadata, IfNoneMatch, **kwargs):
        assert IfNoneMatch == "*"
        if Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = (Body.read() if hasattr(Body, "read") else Body, Metadata)
        self.writes.append(Key)
        self.on_put(Key)

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / "prepared"
    path.mkdir()
    for name in ARTIFACTS:
        (path / name).write_bytes(name.encode())
    manifest = {"tenant": "VND", "model_version": "v1.1", "job_id": "RT-1",
                "artifacts": [file_info(path / name) for name in sorted(ARTIFACTS)]}
    return path, manifest


def test_publish_manifest_last_checksums_idempotence_and_no_overwrite(settings, bundle):
    path, manifest = bundle
    s3 = FakeS3()
    models = Models(settings, s3)
    models.publish(manifest, path)
    assert len(s3.writes) == 5 and s3.writes[-1].endswith("manifest.json")
    models.publish(manifest, path)
    assert len(s3.writes) == 5
    with pytest.raises(JobError, match="TARGET_VERSION_ALREADY_EXISTS"):
        models.publish(dict(manifest, job_id="RT-other"), path)


def test_cancel_during_publish_rolls_back_only_owned_objects(settings, bundle):
    path, manifest = bundle
    s3 = FakeS3()
    models = Models(settings, s3)
    cancelled = False
    def after_put(key):
        nonlocal cancelled
        cancelled = True
    def check():
        if cancelled:
            raise Cancelled()
    s3.on_put = after_put
    with pytest.raises(Cancelled):
        models.publish(manifest, path, check)
    assert len(s3.objects) == 1
    protected = version_prefix("VND", "v1.0") + "lgbm_clf.pkl"
    s3.objects[protected] = (b"base", {"job-id": "BASE"})
    models.rollback("VND", "v1.1", "RT-1")
    assert list(s3.objects) == [protected]


def test_cancel_after_manifest_put_cleans_manifest_too(settings, bundle):
    path, manifest = bundle
    s3 = FakeS3()
    models = Models(settings, s3)
    def check():
        if any(k.endswith("manifest.json") for k in s3.objects):
            raise Cancelled()
    with pytest.raises(Cancelled):
        models.publish(manifest, path, check)
    models.rollback("VND", "v1.1", "RT-1")
    assert not s3.objects


def test_partial_publish_recovery_checks_owner(settings, bundle):
    path, manifest = bundle
    s3 = FakeS3()
    models = Models(settings, s3)
    key = version_prefix("VND", "v1.1") + "cat_encoder.pkl"
    s3.objects[key] = ((path / "cat_encoder.pkl").read_bytes(), {"job-id": "RT-other"})
    with pytest.raises(JobError, match="OWNED_BY_OTHER_JOB"):
        models.publish(manifest, path)
    models.rollback("VND", "v1.1", "RT-1")
    assert key in s3.objects


def test_partial_publish_resumes_without_replacing_existing_bytes(settings, bundle):
    path, manifest = bundle
    s3 = FakeS3()
    models = Models(settings, s3)
    key = version_prefix("VND", "v1.1") + "cat_encoder.pkl"
    s3.objects[key] = ((path / "cat_encoder.pkl").read_bytes(), {"job-id": "RT-1"})
    models.publish(manifest, path)
    assert len(s3.objects) == 5 and key not in s3.writes


def test_lost_response_after_manifest_commit_can_be_reconciled(settings, bundle):
    path, manifest = bundle
    s3 = FakeS3()
    models = Models(settings, s3)
    def lose_response(key):
        if key.endswith("manifest.json"):
            raise ClientError({"Error": {"Code": "InternalError"}}, "PutObject")
    s3.on_put = lose_response
    with pytest.raises(ClientError):
        models.publish(manifest, path)
    assert len(s3.objects) == 5
    s3.on_put = lambda key: None
    models.publish(manifest, path)
    assert len(s3.writes) == 5
