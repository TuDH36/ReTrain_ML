import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    state_dir: Path = Path("./state")
    api_token: str = ""
    callback_url: str = ""
    callback_token: str = ""
    callback_enabled: bool = False
    callback_timeout: float = 10
    callback_retry_max_seconds: float = 60
    max_concurrent_jobs: int = 2
    max_pending_jobs: int = 50
    max_alerts: int = 200000
    train_threads: int = 2
    cancel_grace_seconds: float = 5
    max_train_seconds: float = 14400
    es_url: str = ""
    es_username: str = ""
    es_password: str = ""
    es_api_key: str = ""
    es_verify_tls: bool = True
    es_ca_bundle: str = ""
    es_alert_index: str = "soar_alerts"
    es_event_index: str = "case_events"
    es_tenant_field: str = "customerCode"
    es_label_field: str = "rawData.ai_training.label"
    es_positive_label: str = "TRUE_POSITIVE"
    es_negative_label: str = "FALSE_POSITIVE"
    es_event_link_source: str = "id"
    es_event_link_field: str = "alertId.keyword"
    es_event_data_field: str = "event"
    es_routing: str = ""
    es_batch_size: int = 500
    es_mapping_confirmed: bool = False
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "soar-models"
    minio_region: str = "us-east-1"
    minio_verify_tls: bool = True
    minio_ca_bundle: str = ""
    io_timeout: float = 30

    @property
    def db_path(self):
        return self.state_dir / "jobs.sqlite3"

    @classmethod
    def from_env(cls):
        load_dotenv()
        values = {}
        for name, field in cls.__dataclass_fields__.items():
            raw = os.getenv(name.upper())
            if raw is None:
                continue
            typ = field.type
            if typ is bool:
                if raw.lower() not in {"true", "false"}:
                    raise ValueError(f"{name.upper()} must be true or false")
                values[name] = raw.lower() == "true"
            elif isinstance(field.default, Path):
                values[name] = Path(raw)
            else:
                values[name] = typ(raw)
        return cls(**values)

    def validate(self):
        for name in ("api_token", "es_url", "minio_endpoint",
                     "minio_access_key", "minio_secret_key"):
            if not getattr(self, name) or getattr(self, name).startswith("CHANGE_ME"):
                raise ValueError(f"Configure {name.upper()} in .env")
        if len(self.api_token) < 24:
            raise ValueError("API_TOKEN must contain at least 24 characters")
        if self.callback_enabled:
            if not self.callback_url or not self.callback_token or self.callback_token.startswith("CHANGE_ME"):
                raise ValueError("Configure CALLBACK_URL and CALLBACK_TOKEN before enabling callbacks")
            if "{" in self.callback_url or "}" in self.callback_url:
                raise ValueError("CALLBACK_URL must be a fixed URL; job_id is sent in the JSON body")
        urls = [self.es_url, self.minio_endpoint]
        if self.callback_enabled:
            urls.append(self.callback_url)
        for value in urls:
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Service URLs must use http:// or https://")
        for name in ("max_concurrent_jobs", "max_pending_jobs", "max_alerts", "train_threads",
                     "cancel_grace_seconds", "max_train_seconds", "es_batch_size",
                     "callback_timeout", "callback_retry_max_seconds", "io_timeout"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name.upper()} must be positive")
        if self.es_positive_label == self.es_negative_label:
            raise ValueError("Positive and negative labels must differ")
