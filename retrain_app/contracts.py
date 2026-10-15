from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Identifier = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: Identifier
    tenant_id: Identifier
    base_version: Identifier
    target_version: Identifier
    added_alert_ids: list[str] = Field(default_factory=list, max_length=200000)
    removed_alert_ids: list[str] = Field(default_factory=list, max_length=200000)
    created_by: str = Field(default="system", max_length=200)

    @field_validator("job_id", "tenant_id", "base_version", "target_version")
    @classmethod
    def safe_path(cls, value):
        if ".." in value:
            raise ValueError("Identifier cannot contain '..'")
        return value

    @field_validator("tenant_id")
    @classmethod
    def tenant(cls, value):
        return value.upper()

    @field_validator("added_alert_ids", "removed_alert_ids")
    @classmethod
    def ids(cls, values):
        if any(not v or len(v) > 512 or v != v.strip() for v in values):
            raise ValueError("Alert IDs must be nonempty ES _ids without surrounding whitespace")
        if len(values) != len(set(values)):
            raise ValueError("Duplicate alert IDs")
        return sorted(values)

    @model_validator(mode="after")
    def consistent(self):
        if self.base_version == self.target_version:
            raise ValueError("target_version must differ from base_version")
        if set(self.added_alert_ids) & set(self.removed_alert_ids):
            raise ValueError("Cannot both add and remove the same alert")
        return self


class CancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: Identifier
    tenant_id: Identifier

    @field_validator("tenant_id")
    @classmethod
    def tenant(cls, value):
        return value.upper()


class CancelResponse(BaseModel):
    job_id: str
    tenant_id: str
    status: Literal["CANCELLING", "CANCEL_FAILED", "CANCELLED"]
    error_code: str | None = None


class JobError(Exception):
    """Safe error code, with no raw alert values or credentials."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class Cancelled(Exception):
    pass
