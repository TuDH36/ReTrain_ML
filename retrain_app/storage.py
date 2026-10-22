import hashlib
import json
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .contracts import JobError
from .labels import validate_training_labels

ARTIFACTS = {"cat_encoder.pkl", "tfidf_char.pkl", "tfidf_word.pkl", "lgbm_clf.pkl"}


def file_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def version_prefix(tenant, version):
    return f"tenants/{tenant}/versions/{version}/"


class Models:
    def __init__(self, settings, client=None):
        self.bucket = settings.minio_bucket
        self.client = client or boto3.client(
            "s3", endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_access_key, aws_secret_access_key=settings.minio_secret_key,
            region_name=settings.minio_region, verify=settings.minio_ca_bundle or settings.minio_verify_tls,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          connect_timeout=settings.io_timeout, read_timeout=settings.io_timeout,
                          retries={"max_attempts": 3, "mode": "standard"}),
        )

    def read_manifest(self, tenant, version, optional=False):
        try:
            result = self.client.get_object(Bucket=self.bucket, Key=version_prefix(tenant, version) + "manifest.json")
        except ClientError as exc:
            if optional and exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return None
            raise
        stream = result["Body"]
        try:
            if result["ContentLength"] > 128 * 1024 * 1024:
                raise JobError("MANIFEST_TOO_LARGE")
            manifest = json.loads(stream.read())
        finally:
            stream.close()
        if not isinstance(manifest, dict) or manifest.get("tenant", manifest.get("tenant_id")) != tenant or manifest.get("model_version") != version:
            raise JobError("MANIFEST_IDENTITY_MISMATCH")
        return manifest

    @staticmethod
    def validate_base(manifest):
        if manifest.get("is_test_fixture"):
            raise JobError("TEST_FIXTURE_NOT_TRAINABLE")
        ids = manifest.get("training_alert_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise JobError("BASE_TRAINING_IDS_MISSING_OR_INVALID")
        validate_training_labels(manifest)
        entries = manifest.get("artifacts", [])
        if not isinstance(entries, list) or len(entries) != 4 or {e.get("file") for e in entries} != ARTIFACTS:
            raise JobError("BASE_ARTIFACTS_INVALID")
        for entry in entries:
            sha = entry.get("sha256", "")
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha) or not isinstance(entry.get("size_bytes"), int) or not 0 < entry["size_bytes"] <= 512 * 1024 * 1024:
                raise JobError("BASE_CHECKSUM_OR_SIZE_INVALID")
        cols = manifest.get("text_cols")
        if not isinstance(cols, list) or not cols or any(not isinstance(c, str) or not c.startswith("events.") for c in cols) or len(set(cols)) != len(cols):
            raise JobError("BASE_TEXT_COLUMNS_INVALID")

    def download_base(self, tenant, version, manifest, directory):
        self.validate_base(manifest)
        directory.mkdir(parents=True, exist_ok=True)
        for entry in manifest["artifacts"]:
            path = directory / entry["file"]
            result = self.client.get_object(Bucket=self.bucket, Key=version_prefix(tenant, version) + path.name)
            stream = result["Body"]
            try:
                if result["ContentLength"] != entry["size_bytes"]:
                    raise JobError("BASE_ARTIFACT_SIZE_MISMATCH")
                with path.open("wb") as out:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        out.write(block)
            finally:
                stream.close()
            if file_info(path) != entry:
                # Allow future descriptor fields; compare the fields that identify the bytes.
                actual = file_info(path)
                if any(actual[k] != entry[k] for k in actual):
                    raise JobError("BASE_ARTIFACT_CHECKSUM_MISMATCH")

    def verify_remote(self, key, entry, job_id=None):
        result = self.client.get_object(Bucket=self.bucket, Key=key)
        stream = result["Body"]
        try:
            if job_id is not None and result.get("Metadata", {}).get("job-id") != job_id:
                raise JobError("TARGET_OBJECT_OWNED_BY_OTHER_JOB")
            if result["ContentLength"] != entry["size_bytes"]:
                raise JobError("PUBLISHED_SIZE_MISMATCH")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise JobError("PUBLISHED_CHECKSUM_MISMATCH")
        finally:
            stream.close()

    def put_once(self, key, data, job_id, content_type):
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=data,
                                   ContentType=content_type, Metadata={"job-id": job_id}, IfNoneMatch="*")
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}:
                raise

    def publish(self, manifest, directory, check_cancel=lambda: None):
        """Immutable objects, manifest LAST. Safe to repeat after an ambiguous network failure."""
        tenant, version, job_id = manifest["tenant"], manifest["model_version"], manifest["job_id"]
        prefix = version_prefix(tenant, version)
        check_cancel()
        existing = self.read_manifest(tenant, version, optional=True)
        if existing is not None and existing != manifest:
            raise JobError("TARGET_VERSION_ALREADY_EXISTS")
        for entry in manifest["artifacts"]:
            check_cancel()
            path = directory / entry["file"]
            if file_info(path) != entry:
                raise JobError("PREPARED_ARTIFACT_DAMAGED")
            with path.open("rb") as stream:
                self.put_once(prefix + path.name, stream, job_id, "application/octet-stream")
            self.verify_remote(prefix + path.name, entry, job_id)
        check_cancel()
        data = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        self.put_once(prefix + "manifest.json", data, job_id, "application/json")
        if self.read_manifest(tenant, version) != manifest:
            raise JobError("TARGET_VERSION_ALREADY_EXISTS")
        check_cancel()

    def rollback(self, tenant, version, job_id):
        """After the publisher exits, delete only objects owned by the cancelled job."""
        prefix = version_prefix(tenant, version)
        for name in ["manifest.json", *sorted(ARTIFACTS)]:
            key = prefix + name
            try:
                head = self.client.head_object(Bucket=self.bucket, Key=key)
            except ClientError as exc:
                if exc.response["Error"]["Code"] in {"NoSuchKey", "404", "NotFound"}:
                    continue
                raise
            if head.get("Metadata", {}).get("job-id") != job_id:
                continue
            self.client.delete_object(Bucket=self.bucket, Key=key)
            try:
                self.client.head_object(Bucket=self.bucket, Key=key)
            except ClientError as exc:
                if exc.response["Error"]["Code"] in {"NoSuchKey", "404", "NotFound"}:
                    continue
                raise
            raise JobError("CANCEL_CLEANUP_NOT_CONFIRMED")
