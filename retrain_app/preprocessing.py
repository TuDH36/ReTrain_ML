"""
Tiền xử lý giá trị — sao y logic trong Module_Training.ipynb.

Hai bước được dùng lúc train, áp dụng theo đúng thứ tự:

1. clean_data (cell 4):
   - chuẩn hóa giá trị string: lowercase
   - thay mọi chuỗi backslash `\\+` -> `/`
   (phần chuẩn hóa TÊN CỘT được xử lý riêng ở field_mapping.py)

2. clean_noisy_values (cell 26): dùng regex bỏ 4 loại nhiễu
   - timestamp ISO
   - unix timestamp (10-13 chữ số)
   - hash/hex/base64 dài (>=16 ký tự)
   - chuỗi liền mạch dài (>=40 ký tự)
   các giá trị "vô nghĩa" ({-, _, nan, none, ""}) được giữ nguyên.

Hàm clean_value() gộp cả 2 bước, áp cho TỪNG giá trị trường (cả text lẫn cat),
giống hệt khi train áp clean_noisy_values lên toàn bộ cột object.
"""
import re

# ── Regex giống hệt clean_noisy_values trong notebook ───────────────────────
_PAT_LIST = [
    # 1. Timestamp ISO: 2023-01-15T10:30:00.000Z
    re.compile(
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
        r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    ),
    # 2. Unix timestamp: 10-13 chữ số liên tiếp đứng riêng
    re.compile(r"(?<!\d)\d{10,13}(?!\d)"),
    # 3. Hash/hex/base64 thuần: >=16 ký tự
    re.compile(r"[A-Za-z0-9+/=]{16,}"),
    # 4. Chuỗi liền mạch >= 40 ký tự, không có dấu phân cách
    re.compile(r"[A-Za-z0-9+/=]{40,}"),
]

_MEANINGLESS = {"-", "_", "nan", "none", ""}

# Thay mọi chuỗi backslash thành "/" (clean_data)
_BACKSLASH = re.compile(r"\\+")


def _to_str(val):
    """Chuyển value về string một cách 'sạch'.

    JSON có thể đưa số (int/float/bool). Cột categorical lúc train là string,
    nên ta ép về string. Float nguyên (200.0) -> "200" để khớp category dạng "200".
    """
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        return str(int(val)) if val.is_integer() else str(val)
    if isinstance(val, (list, tuple)):
        # gộp list (vd events.src_ip có thể là mảng IP) bằng dấu phẩy + space
        return ", ".join(_to_str(v) for v in val)
    return str(val)


def is_meaningful(val) -> bool:
    """True nếu value mang DỮ LIỆU THẬT (không rỗng, không thuộc tập meaningless).

    Dùng cho cổng ④: đếm số trường quan trọng "có dữ liệu". Định nghĩa khớp với
    notebook (set meaningless = {-, _, nan, none, ""}). Kiểm tra trên giá trị GỐC
    (chỉ lower+strip), không phụ thuộc clean_noisy_values.
    """
    if val is None:
        return False
    s = _to_str(val).strip().lower()
    return s not in _MEANINGLESS


def clean_value(val, replace_with="-"):
    """Tiền xử lý 1 giá trị trường: clean_data + clean_noisy_values (1 cell)."""
    s = _to_str(val)

    # --- clean_data: lowercase + backslash -> "/" ---
    s = s.lower()
    s = _BACKSLASH.sub("/", s)

    # --- clean_noisy_values: skip giá trị vô nghĩa ---
    s = s.strip()
    if s in _MEANINGLESS:
        return s if s else replace_with
    for pat in _PAT_LIST:
        s = pat.sub(replace_with, s)
    return s.strip() or replace_with
