# Báo cáo Kết quả Thực nghiệm T-SCOPE v2 (`512×512`, 21-Shard Paired Evaluation)

**Kho lưu trữ:** `wagur1/tscope-v2`  
**Artifact đóng gói cuối:** [`outputs_kaggle_pool/merged_benchmarks.json`](../outputs_kaggle_pool/merged_benchmarks.json) (`source_manifest_sha256: 1c240c09ba4ae6970e4d347d3774d902e589e28ef1615562a97980a420fa8866`)  
**Ngày cập nhật:** 2026-11-15

---

## 1. Phạm vi kiểm thử và kiểm chứng trọng số

- Tính hợp lệ của bộ kết quả: `schema_version = 2`, `eligibility.final_ready = true`, `complete_shard_variants = 105` (21 shard × 5 biến thể ablation).
- Tập kiểm thử held-out: `1,648` khung hình liên tiếp (`clip_len = 4`, `stride = 4`, resize `bilinear_square` `512×512`):
  - `MOT17-11` (`900` frames): camera cầm tay trong trung tâm thương mại, người đi bộ lớn ở tiền cảnh.
  - `MOT17-13` (`748` frames, bỏ 2 frame cuối `749..750`): camera gắn trên xe buýt chạy, nền dịch chuyển liên tục giữa các frame.
- Cấu hình nén (`libx265`): `preset = medium`, `rate_control = crf` (`CRF ∈ {22, 27, 32, 37}`), `fps = 25`, `aq_mode = 2`, `bframes = 0` (`IPPP`), `qg_size = 16`, lưới `addroi` tối đa `16×16` vùng, `qoffset ∈ [-0.18, +0.14]`, đo bitrate trực tiếp từ payload gói video MP4.
- Kiến trúc và bộ dò: mạng `TemporalSCOPE` (`6,107,226` tham số) đánh giá trên `Faster R-CNN ResNet50-FPN v2` (`FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT`) và `YOLOv8s`, tính dataset-level 101-point pedestrian AP (lọc các hộp `ignore` theo chuẩn MOT17).

### Trọng số Model Soup đã kiểm định trên `MOT17-09` (`release_ckpts/`)

| Variant | Số shard | Danh sách shard | Val Loss (`MOT17-09`) | SHA-256 (prefix) |
|---|---:|---|---:|---|
| `full_tscope` | 5 | `0, 5, 10, 15, 20` | `4.271654` | `7a6f87259fe8...` |
| `spatial_only_scope` | 4 | `1, 6, 11, 16` | `2.883651` | `b6cd87873076...` |
| `wo_ctu_delta_qp` | 4 | `2, 7, 12, 17` | `4.196459` | `3ad9fb5a1c34...` |
| `iwa_gop_qp_reuse` | 4 | `3, 8, 13, 18` | `4.281704` | `63ebeebc84eb...` |
| `proxy_only_zhao` | 4 | `4, 9, 14, 19` | `2.653610` | `b53ce22e88f8...` |

---

## 2. Kết quả tổng hợp toàn tập (`1,648` frames)

### 2.1. So sánh với Anchor `libx265` và Upper Bound (`overall_methods_summary`)

*(Số âm = tiết kiệm bitrate ở cùng AP; số dương = tốn thêm bitrate)*

| Phương pháp | Total BDBR FRCNN AP50 | Total BDBR FRCNN AP50:95 | Total BDBR YOLO AP50 | Total BDBR YOLO AP50:95 |
| --- | ---: | ---: | ---: | ---: |
| `anchor_x265` | `0.00%` | `0.00%` | `0.00%` | `0.00%` |
| `deterministic_blur_ema` | `+33.87%` | `+36.18%` | `+27.52%` | `+34.82%` |
| `spatial_scope_framewise` | `-8.76%` | `-6.80%` | `-11.78%` | `-9.01%` |
| **`tscope_temporal_ctu_qp` (`full_tscope`)** | **`-19.15%`** | **`-16.32%`** | **`-16.90%`** | **`-12.59%`** |
| `oracle_gt_mask_upper_bound` | `-43.82%` | `-44.92%` | `-37.04%` | `-41.82%` |

### 2.2. Kết quả Ablation 5 biến thể (`ablation_variants_summary`)

| Variant | Total FRCNN AP50 | Total FRCNN AP50:95 | Total YOLO AP50 | Total YOLO AP50:95 |
| --- | ---: | ---: | ---: | ---: |
| **`full_tscope`** | **`-19.15%`** | **`-16.32%`** | `-16.90%` | `-12.59%` |
| `iwa_gop_qp_reuse` | `-18.01%` | `-10.68%` | **`-18.81%`** | **`-13.10%`** |
| `wo_ctu_delta_qp` | `-11.24%` | `-12.04%` | `-11.80%` | `-7.63%` |
| `spatial_only_scope` | `-8.76%` | `-6.80%` | `-11.78%` | `-9.01%` |
| `proxy_only_zhao` | `-7.63%` | `-7.07%` | `-7.10%` | `-5.56%` |

### 2.3. Chi tiết từng điểm CRF (`1,648` frames)

| Phương pháp | CRF | Total bpp | I-frame bpp | ROI PSNR (dB) | FRCNN AP50 | FRCNN AP50:95 | YOLO AP50 | YOLO AP50:95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchor_x265` | 22 | `0.6606` | `1.4567` | `32.95` | `56.39%` | `33.08%` | `50.46%` | `29.37%` |
| `anchor_x265` | 27 | `0.3352` | `0.8652` | `29.77` | `53.93%` | `31.70%` | `48.55%` | `28.19%` |
| `anchor_x265` | 32 | `0.1662` | `0.4849` | `26.79` | `49.33%` | `28.68%` | `44.76%` | `25.64%` |
| `anchor_x265` | 37 | `0.0795` | `0.2538` | `23.98` | `39.45%` | `22.42%` | `36.33%` | `20.26%` |
| `spatial_scope_framewise` | 22 | `0.5735` | `1.2630` | `31.99` | `56.83%` | `33.13%` | `50.71%` | `29.45%` |
| `spatial_scope_framewise` | 27 | `0.2982` | `0.7658` | `29.28` | `54.64%` | `31.82%` | `48.75%` | `28.07%` |
| `spatial_scope_framewise` | 32 | `0.1512` | `0.4388` | `26.56` | `49.22%` | `28.36%` | `44.84%` | `25.38%` |
| `spatial_scope_framewise` | 37 | `0.0740` | `0.2349` | `23.90` | `39.79%` | `22.45%` | `36.65%` | `20.16%` |
| `tscope_temporal_ctu_qp` (`full_tscope`) | 22 | `0.5313` | `1.1438` | `31.79` | `56.80%` | `33.24%` | `50.63%` | `29.41%` |
| `tscope_temporal_ctu_qp` (`full_tscope`) | 27 | `0.2765` | `0.6918` | `29.07` | `54.70%` | `31.76%` | `48.69%` | `28.13%` |
| `tscope_temporal_ctu_qp` (`full_tscope`) | 32 | `0.1398` | `0.3944` | `26.36` | `49.93%` | `28.59%` | `44.76%` | `25.31%` |
| `tscope_temporal_ctu_qp` (`full_tscope`) | 37 | `0.0681` | `0.2105` | `23.74` | `40.22%` | `22.20%` | `36.41%` | `19.79%` |
| `oracle_gt_mask_upper_bound` | 22 | `0.2996` | `0.6678` | `32.42` | `56.89%` | `33.05%` | `49.08%` | `28.68%` |
| `oracle_gt_mask_upper_bound` | 27 | `0.1627` | `0.4066` | `29.21` | `54.02%` | `31.21%` | `47.25%` | `27.42%` |
| `oracle_gt_mask_upper_bound` | 32 | `0.0850` | `0.2350` | `26.18` | `47.76%` | `27.71%` | `43.26%` | `24.87%` |
| `oracle_gt_mask_upper_bound` | 37 | `0.0434` | `0.1307` | `23.41` | `37.17%` | `21.62%` | `34.39%` | `19.62%` |

Đồ thị đường rate-accuracy và cột ablation nằm tại [`fig_bdbr_curves.png`](./figures/fig_bdbr_curves.png), [`fig_ablation_bdbr.png`](./figures/fig_ablation_bdbr.png) và ảnh kiểm tra pre-edit tại [`ablation_preedit_comparison_grid.png`](./figures/ablation_preedit_comparison_grid.png).

---

## 3. Phân rã theo chuỗi (`MOT17-11` và `MOT17-13`) và đối chiếu v1 -> v2

| Chuỗi (Số frame) | Phương pháp / Biến thể | Total FRCNN AP50 | Total FRCNN AP50:95 | Total YOLO AP50 | Total YOLO AP50:95 | ROI PSNR @ CRF22 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **`MOT17-11`** (`900`) | `full_tscope` | `-17.97%` | `-4.91%` | `+213.93%` * | `-14.14%` | `31.94 dB` |
| **`MOT17-11`** (`900`) | `spatial_only_scope` | `-16.37%` | `+1.47%` | `-13.24%` | `-15.87%` | `32.15 dB` |
| **`MOT17-11`** (`900`) | `wo_ctu_delta_qp` | **`-26.62%`** | **`-15.70%`** | **`-22.37%`** | `-14.80%` | `32.18 dB` |
| **`MOT17-11`** (`900`) | `iwa_gop_qp_reuse` | `-25.40%` | `+5.47%` | `+18.19%` | **`-20.13%`** | `31.80 dB` |
| **`MOT17-11`** (`900`) | `proxy_only_zhao` | `+3.17%` | `-3.05%` | `-6.25%` | `-9.64%` | `32.42 dB` |
| **`MOT17-11`** (`900`) | `deterministic_blur_ema` | `+1558.81%` * | `+50.55%` | `+71.66%` | `+49.48%` | `28.59 dB` |
| **`MOT17-11`** (`900`) | `oracle_gt_mask_upper_bound` | `-48.35%` | `-44.56%` | `-18.11%` | `-45.17%` | `32.73 dB` |
| **`MOT17-13`** (`748`) | **`full_tscope`** | **`-16.34%`** | **`-16.22%`** | **`-14.87%`** | **`-15.05%`** | `31.61 dB` |
| **`MOT17-13`** (`748`) | `spatial_only_scope` | `-5.40%` | `-5.49%` | `-8.47%` | `-7.32%` | `31.79 dB` |
| **`MOT17-13`** (`748`) | `wo_ctu_delta_qp` | `-5.79%` | `-6.54%` | `-9.44%` | `-7.10%` | `31.95 dB` |
| **`MOT17-13`** (`748`) | `iwa_gop_qp_reuse` | `-9.46%` | `-7.48%` | `-13.64%` | `-10.61%` | `31.34 dB` |
| **`MOT17-13`** (`748`) | `proxy_only_zhao` | `-2.26%` | `-3.17%` | `-5.18%` | `-4.05%` | `32.06 dB` |
| **`MOT17-13`** (`748`) | `deterministic_blur_ema` | `+41.88%` | `+47.06%` | `+40.40%` | `+49.33%` | `27.47 dB` |
| **`MOT17-13`** (`748`) | `oracle_gt_mask_upper_bound` | `-53.02%` | `-52.49%` | `-46.46%` | `-45.34%` | `32.06 dB` |

*\*Giải thích ngoại lệ BDBR trên `MOT17-11`: ở thước đo `YOLOv8s AP@50`, `MOT17-11` gần như đi ngang từ CRF 22 xuống CRF 32 (`anchor_x265`: `62.49% / 61.76% / 60.04% / 55.21%` tại `0.6851 / 0.3436 / 0.1673 / 0.0790 bpp`; `full_tscope`: `62.34% / 62.05% / 60.20% / 55.03%` tại `0.5401 / 0.2788 / 0.1391 / 0.0675 bpp`). Dù `full_tscope` giảm `14.6%` đến `21.2%` bitrate ở cả 4 mức CRF, đỉnh `62.34%` nằm thấp hơn `anchor_x265` `0.15` điểm nên nội suy spline bậc 3 đơn chuỗi bị nổ biên lên `+213.93%` (`FR50` trên `deterministic_blur_ema` cũng gặp hiện tượng tương tự).*

### Những điểm v2 đã giải quyết so với v1 (`wagur1/research_gap`)

1. Khắc phục chuỗi camera xe buýt `MOT17-13`: trong v1, `full_tscope` gần như mất hết hiệu quả trên `MOT17-13` (`-4.07%` FR50, `-0.22%` FR50:95, thua xa `spatial_only_scope` `-8.17% / -7.17%`). Sang v2, nhờ 6-DOF Cauchy-IRLS affine motion compensation kèm cổng che khuất nền và Static Identity Snap (`< 0.75 px`), `full_tscope` đạt **`-16.34%` FR50 / `-16.22%` FR50:95 / `-14.87%` YOLO50 / `-15.05%` YOLO50:95**, bỏ xa `spatial_only_scope` (`-5.40% / -5.49%`) gần 11 điểm phần trăm trên Faster R-CNN.
2. Bảo toàn tiền cảnh (`ROI PSNR` và detection AP): trên toàn tập 1,648 frame ở CRF 22, `ROI PSNR` của `full_tscope` tăng `+3.46 dB` so với v1 (`28.33 dB -> 31.79 dB`, sát mức `32.95 dB` của `anchor_x265`). Quan trọng hơn, ở cả 4 mức CRF `22 / 27 / 32 / 37`, `FRCNN AP@50` của `full_tscope` (`56.80% / 54.70% / 49.93% / 40.22%`) đều ngang hoặc nhỉnh hơn `anchor_x265` (`56.39% / 53.93% / 49.33% / 39.45%`) trong khi bitrate giảm `14.3%` đến `19.6%`.
3. Thứ hạng tổng thể giữa các ablation: `full_tscope` đứng đầu toàn bộ biến thể triển khai được trên Faster R-CNN (`-19.15%` AP50, `-16.32%` AP50:95 so với `-15.52% / -11.52%` ở v1), vượt cả `wo_ctu_delta_qp` (`-11.24% / -12.04%`) và `spatial_only_scope` (`-8.76% / -6.80%`). Trên YOLOv8s, `iwa_gop_qp_reuse` nhỉnh hơn nhẹ (`-18.81% / -13.10%` so với `-16.90% / -12.59%`).

### Điểm còn hạn chế

- Khoảng cách tới Oracle vẫn lớn (`-19.15%` so với `-43.82%` FR50): `full_tscope` mới cắt được khoảng `19.6%` tổng bpp ở CRF 22 (`0.5313` vs `0.6606`), còn `oracle_gt_mask_upper_bound` giảm tới `54.6%` (`0.2996` bpp). Bản đồ chú ý dự đoán vẫn giữ nhiều vùng nền để bảo toàn recall cho detector.
- Độ mượt đường cong trên `MOT17-11` và độ nhạy ở ngưỡng chặt: ở AP50:95 trên `MOT17-11`, `full_tscope` mới đạt `-4.91%` FR50:95 và tụt nhẹ ở CRF 37 (`37.05%` vs anchor `37.85%`).
- Tốc độ chạy (`FPS`) trên v2 chưa đo lại: bổ sung `estimate_global_affine_irls` và cơ chế carry-forward `dQP` làm thay đổi chi phí suy luận so với bản trước, nên repo chưa công bố bảng FPS/VRAM cho cấu hình v2 `6,107,226` tham số.
