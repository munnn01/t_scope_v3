# T-SCOPE v3

Tiền biên tập video có bù chuyển động và phân bổ lượng tử hóa QG cho `libx265`.
V3 kiểm chứng **temporal processing × semantic ΔQP** với thí nghiệm ghép cặp.

**Trạng thái:** mã và protocol v3; chưa có benchmark v3 hoàn chỉnh. Các số liệu,
hình và trọng số trong `docs/V2_BENCHMARK_REPORT.md`, `outputs_kaggle_pool/` và
`release_ckpts/` thuộc **v2**, không phải kết quả v3. Tài liệu cũ ở
[V2_README.md](docs/V2_README.md).

Đợt train đầu gặp lỗi xử lý overflow AMP sau khi đã vào RUNNING. Bản sửa cho
GradScaler giảm scale và bỏ qua update lỗi; cả sáu nhánh đã vượt canary 128 update,
checkpoint tải về đã được kiểm tra độc lập. Lượt chạy lại dùng mã `92d778c`.
Xem [chẩn đoán và khôi phục](docs/V3_AMP_RECOVERY.md).

## Kế thừa và ma trận thí nghiệm

SCOPE (Dani, Galteri, Bertini) cung cấp conditioning, FiLM U-Net, attention trộn
vật thể/nền và giám sát detector đóng băng. V3 nghiên cứu phần mở rộng: affine
motion compensation, ConvGRU, nền EMA và attention điều khiển QG `quantOffsets`
qua `addroi` trong x265 thật. Xem [nguồn](THIRD_PARTY_NOTICES.md) và
[protocol nghiên cứu](docs/V3_RESEARCH_PROTOCOL.md).

| Variant | Temporal package | Semantic ΔQP | Codec khi train |
|---|---:|---:|---|
| `spatial` | Tắt | Tắt | x265 forward / proxy backward |
| `spatial_qp` | Tắt | Bật | x265 forward / proxy backward |
| `temporal` | Bật | Tắt | x265 forward / proxy backward |
| `temporal_qp` | Bật | Bật | x265 forward / proxy backward |
| `temporal_qp_proxy` | Bật | Bật | Chỉ proxy |
| `scope_paper_adapted` | Tắt | Tắt | Chỉ proxy; báo cáo riêng |

Bốn hàng đầu dùng cùng backbone, dữ liệu và ngân sách. Temporal là gói gồm bù
chuyển động, recurrence, EMA và loss thời gian. Đối chứng proxy chỉ thay đường
codec khi train. Reference image-only là bản viết theo bài báo, thích ứng
MOT17/x265 và dùng objective chung của repo; không phải mã/trọng số chính thức
hoặc bản tái lập kết quả COCO của SCOPE.

Seed `2026 / 2027 / 2028`: **18 lần train độc lập**. Mỗi tài khoản là một nơi
chạy; mỗi job thấy toàn bộ split train. Không trộn trọng số qua seed/model soup.

## Môi trường

Python **3.11 hoặc 3.12**; PyTorch/torchvision **2.5.1 / 0.20.1**; FFmpeg có
`libx265`, `addroi` và `ffprobe`. Chọn wheel CUDA phù hợp driver; CPU có thể cài
cặp torch/torchvision từ `https://download.pytorch.org/whl/cpu` trước.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
ffmpeg -version
ffprobe -version
```

Codec là công cụ hệ thống, không nằm trong requirements Python. Test cần codec
thật hoặc cache trọng số detector có thể được bỏ qua trên máy offline.

## Train và tiếp tục cục bộ

MOT17: train `02/04/05/10`, validation `09`, test `11/13`. Không dùng bản SDP/DPM
trùng video. Thiếu chuỗi/dữ liệu hoặc pretrained detector sẽ dừng, không thay bằng
dữ liệu giả. Faster R-CNN R50-FPN v2 là teacher; YOLO chỉ dùng khi eval. Chỉ claim x265.

```powershell
python -c "import json; from pathlib import Path; from scripts.run_v3 import make_job; Path('outputs_v3').mkdir(exist_ok=True); Path('outputs_v3/job.json').write_text(json.dumps(make_job('temporal_qp',2026),indent=2),encoding='utf-8')"
python scripts/run_v3.py --mode train --job-file outputs_v3/job.json --data-root D:/datasets/MOT17/train --output-dir outputs_v3/local
```

Mặc định: 512×512, clip 4 frame, batch 2, tích lũy gradient 2, tối đa 36 epoch,
patience 10, ít nhất 12 epoch, ngân sách phiên 7,8 giờ. Crop/severity phụ thuộc
seed/epoch/clip, độc lập model và account. `.last.pt` chứa model, optimizer, scaler,
vị trí tiếp tục, best weights, history và provenance. Hết thời gian được đánh dấu
`time_budget_exhausted`, không coi là hội tụ. Tiếp tục cùng job:

```powershell
python scripts/run_v3.py --mode train --job-file outputs_v3/job.json --data-root D:/datasets/MOT17/train --output-dir outputs_v3/local --resume outputs_v3/local/temporal_qp_seed2026.last.pt
```

## Nhiều tài khoản Kaggle

Pool JSON mapping `{username: access_token}` nằm **ngoài repo**. Không commit pool
hoặc đưa token vào worker/manifest. Công cụ hỗ trợ modern Kaggle access token.

```powershell
python -m pip install -r requirements-kaggle.txt
python scripts/kaggle_v3.py --action verify --pool-file D:/private/pool.json
$revision = git rev-parse HEAD
```

Đẩy commit đó lên GitHub trước dispatch. Worker checkout đúng SHA và kiểm tra
nguồn sạch. Worker tạo môi trường Python 3.11.14 riêng bằng uv 0.9.9, kể cả khi
Python mặc định của Kaggle là 3.13; cặp torch/torchvision vẫn giữ nguyên.
Chạy canary với ΔQP động trước; xác nhận optimizer updates và checkpoint:

```powershell
python scripts/kaggle_v3.py --action push --pool-file D:/private/pool.json --manifest outputs_v3/canary_manifest.json --run-id v3-canary --revision $revision --canary-only --only-jobs temporal_qp_seed2026
python scripts/kaggle_v3.py --action status --pool-file D:/private/pool.json --manifest outputs_v3/canary_manifest.json
python scripts/kaggle_v3.py --action pull --pool-file D:/private/pool.json --manifest outputs_v3/canary_manifest.json
```

Sau khi canary thành công, chạy ma trận đầy đủ:

```powershell
python scripts/kaggle_v3.py --action plan --pool-file D:/private/pool.json --manifest outputs_v3/train_manifest.json --run-id v3-train --revision $revision
python scripts/kaggle_v3.py --action push --pool-file D:/private/pool.json --manifest outputs_v3/train_manifest.json
python scripts/kaggle_v3.py --action status --pool-file D:/private/pool.json --manifest outputs_v3/train_manifest.json
python scripts/kaggle_v3.py --action pull --pool-file D:/private/pool.json --manifest outputs_v3/train_manifest.json
```

Giới hạn bằng `--max-accounts`/`--only-jobs`. Thiếu account đủ quota thì job giữ
pending, không đổi danh tính variant/seed. Không ghi đè kernel có sẵn. Manifest
ghi job ID, SHA, username, worker hash và trạng thái, không chứa credentials.

## Đánh giá

Chỉ nhận best weights từ job hoàn thành/hội tụ. Test ghép cặp 1.648 frame held-out,
IPPP, CRF `{22,27,32,37}` và cùng physical frame IDs cho anchor/model:

```powershell
python -m pip install -r requirements-eval.txt
python scripts/run_v3.py --mode eval --job-file outputs_v3/job.json --checkpoint outputs_v3/local/temporal_qp_seed2026.last.pt --data-root D:/datasets/MOT17/train --output-dir outputs_v3/eval --eval-yolo
python scripts/merge_v3.py --input-dir outputs_v3/eval --output outputs_v3/benchmark_v3.json
```

Lưu raw detections, packet sizes/types và hash dữ liệu. AP tính toàn tập; bitrate
lấy payload video, loại MP4 container/global headers. Chính sách pedestrian/ignore
được khai báo riêng, không tương đương đầy đủ COCO crowd/area/maxDets.
**PCHIP là BDBR chính; cubic kiểm tra độ nhạy.** AP không đơn điệu/plateau/không
có miền giao được ghi unavailable, không thêm nhiễu hoặc ngoại suy. Báo cáo từng
seed và mean/std; final-ready chỉ bật khi đủ job, dữ liệu ghép cặp và provenance.

Mã chính: `v3_protocol.py`, `run_v3.py`, `kaggle_v3.py`, `rd_metrics.py`,
`merge_v3.py` trong `scripts/`; reference ở `models/scope_reference.py`.

Canary GPU thật và CI đã được kiểm chứng; xem [bằng chứng](docs/V3_VALIDATION.md).
Khi resume/eval, dùng checkout đúng `source_revision` trong job JSON. Commit cập
nhật tài liệu sau khi dispatch không thay đổi mã nguồn mà worker đã ghim.
