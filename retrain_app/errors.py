"""Four public training errors, independent of internal diagnostic reasons."""

TRAINING_ERROR_MESSAGES = {
    "DATA_INVALID": (
        "Dữ liệu cảnh báo không đủ hoặc không hợp lệ để huấn luyện. "
        "Vui lòng kiểm tra các cảnh báo đã chọn, nhãn và dữ liệu sự kiện."
    ),
    "MODEL_INVALID": (
        "Mô hình, cấu hình hoặc phiên bản đã chọn không hợp lệ. "
        "Vui lòng kiểm tra phiên bản nền và phiên bản đích."
    ),
    "STORAGE_ERROR": (
        "Không thể đọc hoặc lưu dữ liệu trên Elasticsearch/MinIO. "
        "Vui lòng kiểm tra kết nối, quyền truy cập và dịch vụ lưu trữ trước khi thử lại."
    ),
    "TRAINING_FAILED": (
        "Quá trình huấn luyện bị gián đoạn hoặc gặp lỗi xử lý nội bộ. "
        "Vui lòng thử lại; nếu lỗi tiếp diễn, liên hệ quản trị viên kèm mã job."
    ),
}

_DATA_ERRORS = {
    "ES_TRAINING_LABEL_INVALID", "ES_ALERT_NOT_FOUND", "ES_ALERT_TENANT_MISMATCH",
    "ES_ALERT_EVENTS_MISSING", "ES_EVENT_REFERENCE_MISSING_OR_NONUNIQUE",
    "ES_EVENT_LINK_MISMATCH", "ES_EVENT_JSON_INVALID", "ES_EVENT_OBJECT_REQUIRED",
    "TRAINING_ALERT_NOT_SELECTED", "REMOVED_IDS_NOT_IN_BASE", "EMPTY_SELECTED_DATASET",
    "TOO_MANY_TRAINING_ALERTS", "NO_USABLE_FEATURE_ROWS",
    "TRAINING_REQUIRES_BOTH_TP_AND_FP_AFTER_FILTERING",
}
_MODEL_ERRORS = {
    "BASE_TRAINING_LABELS_MISSING_OR_INVALID", "BASE_TRAINING_IDS_MISSING_OR_INVALID",
    "TEST_FIXTURE_NOT_TRAINABLE", "MANIFEST_IDENTITY_MISMATCH", "MANIFEST_TOO_LARGE",
    "BASE_ARTIFACTS_INVALID", "BASE_CHECKSUM_OR_SIZE_INVALID",
    "BASE_ARTIFACT_SIZE_MISMATCH", "BASE_ARTIFACT_CHECKSUM_MISMATCH",
    "BASE_TEXT_COLUMNS_INVALID", "BASE_CATEGORICAL_COLUMNS_MISSING",
    "SKLEARN_VERSION_MISMATCH", "FEATURES_MUST_COME_FROM_EVENTS", "INVALID_FEATURE_ALIAS",
    "INVALID_THRESHOLD", "TARGET_VERSION_ALREADY_EXISTS", "TARGET_OBJECT_OWNED_BY_OTHER_JOB",
}
_STORAGE_ERRORS = {
    "ES_PARTIAL_RESULT", "ES_INEXACT_TOTAL", "ES_DUPLICATE_EVENT_PAGE",
    "ES_SCROLL_ID_MISSING", "ES_INCOMPLETE_ALERTS", "ES_INCOMPLETE_EVENTS",
    "PUBLISHED_SIZE_MISMATCH", "PUBLISHED_CHECKSUM_MISMATCH",
    "HTTPERROR", "CONNECTIONERROR", "CONNECTTIMEOUT", "READTIMEOUT", "TIMEOUT",
    "SSLERROR", "PROXYERROR", "CHUNKEDENCODINGERROR", "CONTENTDECODINGERROR",
    "CLIENTERROR", "ENDPOINTCONNECTIONERROR", "CONNECTTIMEOUTERROR", "READTIMEOUTERROR",
    "NOCREDENTIALSERROR", "PARTIALCREDENTIALSERROR", "CREDENTIALRETRIEVALERROR",
    "CONNECTIONCLOSEDERROR", "HTTPCLIENTERROR", "INCOMPLETEREADERROR", "RESPONSESTREAMINGERROR",
}

# Safe explanations for common causes. Never return raw exception text, IDs,
# alert contents, credentials, or library-specific diagnostics to the user.
_REASON_MESSAGES = {
    "ES_TRAINING_LABEL_INVALID": (
        "Có cảnh báo thiếu nhãn hoặc nhãn không hợp lệ. "
        "Cảnh báo bổ sung cần có aiLabel là true positive hoặc false positive; "
        "dữ liệu kế thừa cần có nhãn chuẩn đã học."
    ),
    "ES_ALERT_NOT_FOUND": "Không tìm thấy hoặc không đọc được một số cảnh báo đã chọn trên Elasticsearch.",
    "ES_ALERT_TENANT_MISMATCH": "Danh sách huấn luyện có cảnh báo không thuộc khách hàng đang chọn.",
    "ES_ALERT_EVENTS_MISSING": "Có cảnh báo thiếu dữ liệu sự kiện cần dùng để huấn luyện.",
    "NO_USABLE_FEATURE_ROWS": "Không còn cảnh báo nào có đặc trưng sử dụng được sau khi làm sạch dữ liệu.",
    "TRAINING_REQUIRES_BOTH_TP_AND_FP_AFTER_FILTERING": (
        "Sau khi lọc dữ liệu và loại nhóm xung đột nhãn, tập huấn luyện không còn đủ cả TP và FP. "
        "Vui lòng bổ sung cảnh báo có nhãn phù hợp."
    ),
    "TEST_FIXTURE_NOT_TRAINABLE": (
        "Phiên bản nền là model giả lập dùng để kiểm thử, không dùng được để huấn luyện. "
        "Vui lòng chọn phiên bản có bộ model thật."
    ),
    "SKLEARN_VERSION_MISMATCH": "Phiên bản thư viện của worker không tương thích với mô hình nền. Cần kiểm tra môi trường triển khai.",
    "TARGET_VERSION_ALREADY_EXISTS": "Phiên bản đích đã tồn tại trên MinIO. Vui lòng sử dụng phiên bản đích khác.",
    "TRAIN_TIMEOUT": "Quá trình huấn luyện vượt thời gian cho phép và đã bị dừng. Vui lòng kiểm tra kích thước dữ liệu và tài nguyên máy chủ.",
    "WORKER_RESTARTED": "Dịch vụ AI đã khởi động lại khi job chưa hoàn tất. Cần tạo lần huấn luyện mới.",
    "SERVICE_STOPPED": "Dịch vụ AI đã bị dừng khi job chưa hoàn tất. Cần tạo lần huấn luyện mới.",
    "MEMORYERROR": "Máy chủ không đủ bộ nhớ để hoàn tất huấn luyện. Vui lòng giảm dữ liệu hoặc bổ sung tài nguyên.",
}


def training_error(reason):
    """Normalize both new failures and older persisted jobs at the API boundary."""
    if not isinstance(reason, str):
        reason = ""
    if reason in TRAINING_ERROR_MESSAGES:
        code = reason
    elif reason in _DATA_ERRORS:
        code = "DATA_INVALID"
    elif reason in _MODEL_ERRORS:
        code = "MODEL_INVALID"
    elif reason in _STORAGE_ERRORS:
        code = "STORAGE_ERROR"
    else:
        code = "TRAINING_FAILED"
    return {"error_code": code,
            "error_message": _REASON_MESSAGES.get(reason, TRAINING_ERROR_MESSAGES[code])}
