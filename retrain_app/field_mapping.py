"""
Map tên trường trong JSON đầu vào (đôi khi "xấu" nhưng đúng) về đúng tên trường
canonical mà model dùng khi train (vd: "events.src_ip").

Vì MỖI TENANT có bộ trường quan trọng khác nhau, lớp `FieldMapper` được khởi tạo
theo từng tenant với:
  - feature_columns : danh sách cột canonical của tenant (text + cat)
  - extra_aliases   : alias thủ công riêng tenant {tên_xấu: tên_canonical}

Cơ chế nhận diện 1 tên trường, theo thứ tự ưu tiên:
  1. exact (chuẩn hóa giống clean_data: lowercase + space->_ + backslash->/,
     kèm biến thể dot->underscore để phân biệt processname vs process_name).
  2. exact sau khi thêm prefix "events." (key gốc trong events[]: "process_name").
  3. compact full (bỏ mọi ký tự không phải a-z0-9). "Events Src IP" -> "eventssrcip".
  4. compact full sau khi thêm prefix "events".
  5. compact suffix (thiếu prefix events.). "Src IP" -> "srcip".
Alias thủ công có ưu tiên cao nhất.

Hỗ trợ JSON phẳng ("events.src_ip": ...) lẫn lồng ({"events": {...}}).
Trường không map được -> bỏ qua (id, hash, time, keyword, ...).
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from .preprocessing import is_meaningful

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
EVENT_PREFIX = "events."


def _compact(name: str) -> str:
    """Bỏ mọi ký tự không phải a-z0-9, lowercase."""
    return _NON_ALNUM.sub("", str(name).lower())


def _clean_data_colname(name: str) -> str:
    """Chuẩn hóa tên cột giống clean_data: lowercase + space->_ + backslash->/."""
    s = "_".join(str(name).lower().split())
    return s.replace("\\", "/")


def flatten_json(payload: Dict[str, Any], parent: str = "") -> Dict[str, Any]:
    """Làm phẳng JSON lồng. {"events": {"src_ip": x}} -> {"events.src_ip": x}."""
    flat: Dict[str, Any] = {}
    for k, v in payload.items():
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            flat.update(flatten_json(v, key))
        else:
            flat[key] = v
    return flat


class FieldMapper:
    """Map tên trường xấu -> canonical cho MỘT tenant cụ thể."""

    def __init__(self, feature_columns: List[str], extra_aliases: Optional[Dict[str, str]] = None):
        self.feature_columns = list(feature_columns)
        self.extra_aliases = extra_aliases or {}
        self._build_index()

    def _build_index(self) -> None:
        self._exact: Dict[str, str] = {}
        self._compact_full: Dict[str, str] = {}
        self._compact_suffix: Dict[str, str] = {}

        for col in self.feature_columns:
            low = col.lower()
            self._exact[_clean_data_colname(col)] = col
            self._exact[low] = col
            # Biến thể dot->underscore: phân biệt "events_process_name" với
            # "events_processname" (schema có thể có cả 2 cột).
            self._exact[low.replace(".", "_")] = col

            self._compact_full.setdefault(_compact(col), col)

            if low.startswith("events."):
                suffix = low[len("events."):]
                self._compact_suffix.setdefault(_compact(suffix), col)

        # alias thủ công ưu tiên cao nhất
        for ugly, canon in self.extra_aliases.items():
            self._exact[_clean_data_colname(ugly)] = canon
            self._compact_full[_compact(ugly)] = canon

    def resolve(self, raw_key: str) -> Optional[str]:
        """Map 1 tên trường -> canonical, hoặc None nếu không thuộc feature."""
        norm = _clean_data_colname(raw_key)
        if norm in self._exact:
            return self._exact[norm]
        if raw_key.lower() in self._exact:
            return self._exact[raw_key.lower()]
        # Key trong events[] là tên trường Splunk gốc, KHÔNG có tiền tố "events." (vd "process_name").
        # Phải thử exact với tiền tố TRƯỚC khi so compact: compact làm trùng "process_name" với
        # "processname", "_sourcetype" với "sourcetype" -> giá trị bị đưa nhầm cột.
        if EVENT_PREFIX + norm in self._exact:
            return self._exact[EVENT_PREFIX + norm]

        comp = _compact(raw_key)
        if comp in self._compact_full:
            return self._compact_full[comp]
        if ("events" + comp) in self._compact_full:
            return self._compact_full["events" + comp]
        if comp in self._compact_suffix:
            return self._compact_suffix[comp]
        return None

    def map_payload(self, flat: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Map dict ĐÃ LÀM PHẲNG -> ({canonical: value}, [keys bị bỏ qua]).

        Khi nhiều key cùng map vào 1 canonical: ưu tiên giá trị có nghĩa (giá trị rỗng/"-"
        không được ghi đè giá trị thật đã có).
        """
        mapped: Dict[str, Any] = {}
        ignored: List[str] = []
        for raw_key, value in flat.items():
            canon = self.resolve(raw_key)
            if canon is None:
                ignored.append(raw_key)
                continue
            if canon in mapped and not is_meaningful(value):
                continue
            mapped[canon] = value
        return mapped, ignored
