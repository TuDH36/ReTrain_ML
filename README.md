# AI Retrain

Dịch vụ độc lập với `Product/api`. BE lấy job từ Redis, gọi AI qua HTTP;
AI không đọc Redis và không truy cập PostgreSQL của BE.

## Chạy trên Ubuntu hoặc Docker Desktop

Clone repository độc lập, hoặc chuyển **thư mục Retrain** sang server. Lần đầu:

```bash
git clone https://github.com/TuDH36/ReTrain_ML.git Retrain
cd Retrain
```

```bash
cp .env.example .env
# Sửa .env theo các mục bên dưới.
docker compose up -d --build
docker compose logs -f retrain
```

Với bản đã clone từ Git, cập nhật source và build lại bằng:

```bash
git pull --ff-only origin main
docker compose up -d --build
```

`.env` và dữ liệu runtime không nằm trong Git. Các bản cập nhật dịch vụ được commit
và push lên nhánh `main` của repository trên.

Windows PowerShell: dùng `Copy-Item .env.example .env` nếu chưa có `.env`.
Mở `http://localhost:8002/docs` để thử API, nhập API_TOKEN bằng nút Authorize.
`GET /health` kiểm tra tiến trình API và số callback còn chờ; không kiểm tra kết nối ES/MinIO.
Server không có Internet: build ở máy có mạng, `docker save soar-ai-retrain:1.0 -o retrain.tar`,
chuyển image rồi `docker load -i retrain.tar` và `docker compose up -d --no-build`.

**Cần cấu hình:** API_TOKEN, CALLBACK_URL/TOKEN của BE, kết nối ES, MinIO và mapping ES.
Dùng địa chỉ truy cập được **từ container**; `127.0.0.1` bên trong container không phải host.
Đổi `BIND_ADDRESS=0.0.0.0` nếu BE gọi từ máy khác. Không đưa mật khẩu vào image.

Mapping mẫu khớp dạng uploader trong repository, chưa khẳng định khớp SOC:

| Biến | Ý nghĩa |
|---|---|
| ES_TENANT_FIELD | Trường trong `_source` xác nhận alert thuộc tenant |
| ES_LABEL_FIELD | Nhãn TP/FP được SOC xác nhận, không phải nhãn AI dự đoán |
| ES_EVENT_LINK_SOURCE | `_id`, `id` hoặc `alertId` của alert để nối event |
| ES_EVENT_LINK_FIELD | Trường keyword/exact-match trong `case_events` |
| ES_EVENT_DATA_FIELD | Object event hoặc chuỗi JSON chứa feature |

Sau khi kiểm tra mapping, đặt `ES_MAPPING_CONFIRMED=true`. Worker chỉ **đọc** ES.
ID nối event phải duy nhất giữa các tenant; mỗi alert được xác minh tenant trước khi lấy event.
Nếu một alert có nhiều event, worker sắp xếp theo `_id` event và lấy giá trị có nghĩa đầu tiên
cho từng trường. BE cần gửi cùng thứ tự đó sang AI Predict để đặc trưng nhất quán.

MinIO cần bucket `soar-models` đã tồn tại và credential có quyền đọc/ghi/xóa object
(quyền xóa dùng để dọn phiên bản khi hủy job).
Model nền phải có 4 artifact, checksum, `text_cols`, `training_alert_ids` thật của đúng ES;
không chấp nhận fixture. Không tự tạo bucket hoặc sửa phiên bản đang áp dụng.

## Hợp đồng với BE

AI yêu cầu `Authorization: Bearer <API_TOKEN>` cho hai API.

### Nhận job

`POST /api/v1/ai/retrain/start`

```json
{
  "job_id": "RT-20261015-01",
  "tenant_id": "VND",
  "base_version": "v1.0",
  "target_version": "v1.2",
  "added_alert_ids": ["ES-document-id-new-1", "ES-document-id-new-2"],
  "removed_alert_ids": ["ES-document-id-in-base"],
  "created_by": "admin5"
}
```

ID là `_id` document alert trên ES, không phải ID Excel hoặc ID event.
BE cấp target_version chưa tồn tại; có thể chọn bất kỳ base_version nào.
AI lưu request bền vững rồi trả `202` ngay:

```json
{
  "job_id": "RT-20261015-01",
  "tenant_id": "VND",
  "target_version": "v1.2",
  "status": "QUEUED",
  "manifest_key": "tenants/VND/versions/v1.2/manifest.json"
}
```

`202` xác nhận đã nhận job, chưa xác nhận dữ liệu ES/model hợp lệ. Kiểm tra sâu chạy bất đồng bộ.
Gửi lại cùng job + payload trả trạng thái hiện tại, không train lần hai (job kết thúc trả `200`).
Job ID bị dùng lại với payload khác, tenant đang bận hoặc target đã đặt trước: `409`.
Sai token: `401`; body không hợp lệ: `422`; thiếu xác nhận mapping/hàng đợi đầy: `503`.
BE retry request lỗi mạng bằng **cùng job_id**, chỉ ACK Redis sau khi AI nhận thành công.

### Callback tiến độ AI → BE

BE triển khai `POST /internal/retrain-jobs/{job_id}/status`, hoặc cấu hình URL khác trong `.env`.

```http
Authorization: Bearer <CALLBACK_TOKEN>
X-Event-Seq: 1
Content-Type: application/json

{"status":"TRAIN"}
```

| Status | BE hiển thị |
|---|---|
| TRAIN | Đọc dữ liệu, tiền xử lý và fit 4 artifact |
| EVALUATE | Tính điểm trên tập train, kiểm tra reload artifact |
| PUBLISH | Đang lưu model và manifest vào MinIO |
| DONE | Model đã lưu và xác minh xong, sẵn sàng để BE áp dụng |
| FAILED | Job lỗi; mã lỗi lưu trong ledger/log của AI |
| CANCELLED | Tiến trình đã dừng hoàn toàn |

Body chỉ có `status`, không có `steps`. BE tự cập nhật bước 1–2 trên giao diện.
BE ghi trạng thái và `last_event_seq` vào DB trước khi trả `2xx`.
Callback gửi lại giữ nguyên sequence; BE bỏ qua event đã nhận và vẫn trả `2xx`.
AI retry có backoff và giữ thứ tự **trong từng job**, kể cả sau restart.
BE giữ `CANCELLING` khi nhận callback TRAIN/EVALUATE cũ đang trên đường gửi;
chỉ mở khóa tenant sau trạng thái cuối, không mở khóa khi mới nhận phản hồi hủy `202`.
Khi DONE, BE đọc manifest_key đã biết để lấy metrics, số lượng và ID đã train.
Nếu cần lấy lỗi chi tiết, BE có thể gửi lại đúng request start cũ để nhận `error_code`.

### Hủy job

`POST /api/v1/ai/retrain/cancel`

```json
{"job_id":"RT-20261015-01","tenant_id":"VND"}
```

QUEUED: trả `200 CANCELLED`. TRAIN/EVALUATE: trả `202 CANCELLING`, tiến trình cha
dừng đúng tiến trình con (TERM, sau thời gian grace thì KILL), đợi exit, dọn dữ liệu job rồi callback CANCELLED.
Gọi hủy lặp lại an toàn. Không tìm thấy job: `404`; BE phải xử lý cuộc đua với request start
đang gửi, không được hiểu `404` là đã hủy. DONE/FAILED: `409 JOB_NOT_CANCELLABLE`.
PUBLISH vẫn hủy được: trả `202 CANCELLING`, chờ request MinIO đang gửi kết thúc, ngừng upload tiếp,
xóa đúng object có metadata `job-id` của job này (manifest trước), xác minh rồi callback CANCELLED.
Không xóa model nền hoặc object của job khác. MinIO lỗi thì giữ CANCELLING và retry dọn dẹp.
Cuộc đua hủy/hoàn tất được chốt bằng transaction: nếu DONE đã ghi trước thì hủy trả 409.
BE không đưa phiên bản của job chưa DONE vào trạng thái sẵn sàng áp dụng, kể cả đã thấy manifest.

## Pipeline và lưu trữ

1. Lấy `(training_alert_ids nền − removed_alert_ids) ∪ added_alert_ids`.
2. Đọc alert/event ES; sai tenant, thiếu ID/event, sai nhãn hoặc truy vấn thiếu dữ liệu → FAILED.
3. Chuẩn hóa như notebook/Predict; loại dòng không có feature có nghĩa và **toàn bộ nhóm
   trùng feature nhưng khác nhãn** trước khi xóa nhiễu regex.
4. Clone cấu hình model nền, fit lại TF-IDF char/word, OrdinalEncoder và LightGBM.
   Fit toàn bộ dữ liệu còn lại, **không validation, không early stopping**, theo yêu cầu đã chốt.
5. Tính Precision/Recall/F1, TP/FP/FN/TN ở ngưỡng 0.5, 0.9 và ngưỡng model; reload kiểm tra.
   Manifest ghi `evaluation.dataset=train`, `is_independent_test=false`.
6. Upload 4 artifact, xác minh SHA256, ghi **manifest cuối cùng** bằng conditional PUT.

```
soar-models/tenants/VND/versions/v1.2/
  cat_encoder.pkl
  tfidf_char.pkl
  tfidf_word.pkl
  lgbm_clf.pkl
  manifest.json
```

`training_alert_ids` chỉ chứa ID thực sự đã fit. `added_count`/`removed_count` là chênh lệch
thực tế với tập nền, gồm cả dòng tự loại; `training` ghi riêng số requested/filter để BE đối chiếu.
Không sinh CSV, Parquet hay dataset snapshot. Dữ liệu là giá trị ES tại lúc job đọc;
nếu SOC sửa nội dung/nhãn về sau, train lại cùng ID có thể cho kết quả khác.

SQLite `/state/jobs.sqlite3` lưu request, trạng thái và callback outbox; volume Docker giữ qua
restart. **Một container, một Uvicorn process**, không scale replica với các volume riêng
(khóa tenant sẽ không còn dùng chung). Điều chỉnh MAX_CONCURRENT_JOBS và TRAIN_THREADS để
train nhiều tenant trong cùng service. Không chạy `docker compose down -v` nếu cần giữ ledger.
Restart giữa TRAIN/EVALUATE: FAILED; giữa CANCELLING: tiếp tục dọn rồi CANCELLED; giữa PUBLISH: tiếp tục upload
từ bộ đã chuẩn bị. Mất mạng khi PUBLISH giữ trạng thái đó và retry, không báo thành công giả.
Các object upload dở chưa có manifest không phải phiên bản READY; BE chỉ liệt kê phiên bản có manifest.
Các trạng thái đang train không có phần trăm giả lập; API không hứa heartbeat theo từng vòng fit.

## Kiểm thử

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests dùng ES/S3 giả lập và dữ liệu tổng hợp, không ghi vào môi trường thật.
Các file `preprocessing.py` và `field_mapping.py` được sao chép độc lập từ Predict để giữ cùng
quy tắc feature; thay đổi sau này cần kiểm tra parity cả hai dịch vụ. Runtime ML được pin
tương thích sklearn 1.3.0 của model nền; chỉ load pickle từ nguồn MinIO tin cậy.
