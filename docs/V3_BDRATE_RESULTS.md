# T-SCOPE v3: thống kê BD-rate và đánh giá kết quả

Ngày tổng hợp: 17/11/2026. Đủ **18/18 training và evaluation**, gồm 6 nhánh × 3 seed độc lập (2026/2027/2028). Tất cả checkpoint dừng theo early stopping (`converged`) và được đối chiếu SHA-256 với đầu vào evaluation.

## Cách đọc và điều kiện đo

**Đơn vị: %. Giá trị âm là tiết kiệm bitrate ở cùng AP.** Mỗi ô là trung bình ± độ lệch chuẩn mẫu của ba seed. Đây là BD-rate theo **AP phát hiện người**, không phải BD-rate theo PSNR.

Anchor: x265 với đầu vào gốc. Test: MOT17-11 và MOT17-13, 1.648 frame vật lý (412 clip), resize 512×512, GOP IPPP dài 4, CRF 22/27/32/37. Tổng bitrate đếm video packet payload, không gồm container/global headers. AP được tính trên toàn bộ test, không lấy trung bình AP từng clip. Faster R-CNN R50-FPN v2 được dùng trong training; YOLOv8s chỉ dùng evaluation.

PCHIP log-rate/AP là cách tính chính, chỉ tích phân miền AP giao nhau; không ngoại suy hoặc sửa đường cong. P-frame BD-rate sử dụng riêng bitrate P-frame và AP toàn bộ clip, vì vậy không phải AP đo riêng trên P-frame. Metric AP dùng chính sách pedestrian/ignore tùy chỉnh, không đồng nhất hoàn toàn với COCO chính thức.

## Bảng chính: AP50–95

| Nhánh | Tổng — Faster R-CNN | Tổng — YOLOv8s | P-frame — Faster R-CNN | P-frame — YOLOv8s | n |
|---|---:|---:|---:|---:|---:|
| `spatial` | -11.16 ± 3.38 | -10.35 ± 4.35 | -8.04 ± 3.88 | -6.82 ± 5.92 | 3 |
| `spatial_qp` | -14.88 ± 5.51 | -11.78 ± 5.67 | -9.95 ± 4.23 | -5.38 ± 5.72 | 3 |
| `temporal` | -9.75 ± 3.43 | -8.57 ± 4.70 | -0.18 ± 6.09 | +1.71 ± 8.14 | 3 |
| `temporal_qp` | -10.82 ± 0.94 | -7.45 ± 2.54 | -4.25 ± 1.66 | +0.77 ± 5.02 | 3 |
| `temporal_qp_proxy` | -10.31 ± 1.40 | -8.94 ± 2.26 | -3.66 ± 2.49 | -1.66 ± 0.97 | 3 |
| `scope_paper_adapted` | -3.11 ± 1.47 | -5.11 ± 1.36 | -1.79 ± 2.26 | -4.57 ± 2.58 | 3 |

## Đánh giá và hướng phát triển

1. **Spatial + ΔQP có trung bình tốt nhất ở bitrate tổng trên cả hai detector.** `spatial_qp` đạt −14,88 ± 5,51% (FRCNN) và −11,78 ± 5,67% (YOLO). Độ phân tán theo seed còn lớn; đây là xếp hạng trung bình quan sát được, không phải kết luận ưu thế có ý nghĩa thống kê.

`temporal_qp` ổn định hơn ở total FRCNN (độ lệch chuẩn 0,94 điểm %), nhưng mức tiết kiệm trung bình thấp hơn spatial + ΔQP. Độ ổn định này chưa đủ để đổi kết luận về hiệu quả trung bình hoặc detector chưa dùng trong training.

2. **Chưa có bằng chứng gói temporal làm tốt hơn spatial trong protocol này.** So sánh trực tiếp `spatial_qp` → `temporal_qp` làm bitrate tổng tăng +5,45 ± 7,13% với FRCNN và +5,29 ± 4,10% với YOLO. Khi không có ΔQP, thêm temporal tăng +1,71 ± 0,56% và +2,00 ± 0,48%. Vì vậy chưa đủ cơ sở lấy hiệu quả temporal làm kết luận chính của nghiên cứu.

3. **ΔQP giúp trung bình bitrate tổng rõ hơn trong nhánh spatial.** So sánh trực tiếp `spatial` → `spatial_qp` đạt −3,84 ± 2,46% (FRCNN) và −1,47 ± 1,75% (YOLO). Tuy nhiên P-frame YOLO tăng +1,69 ± 0,47%, nên không diễn giải lợi ích tổng thành lợi ích inter-frame.

4. **Full v3 tốt hơn reference adapted về bitrate tổng, nhưng chưa phải bằng chứng vượt SCOPE gốc.** `scope_paper_adapted` → `temporal_qp` đạt −7,79 ± 0,46% với FRCNN và −2,48 ± 2,49% với YOLO. Đây chỉ là so sánh trong cấu hình MOT17/x265 chung của dự án.

5. **Real-codec training chưa chứng minh tăng khả năng chuyển sang YOLO so với proxy.** So sánh trực tiếp proxy → real của cùng temporal + ΔQP đạt −0,41 ± 0,53% (FRCNN), nhưng +1,72 ± 3,91% (YOLO) ở bitrate tổng. Tất cả các nhánh vẫn được evaluation bằng x265 thật.

6. **Kết luận P-frame nhạy với nội suy.** Ví dụ YOLO của `temporal_qp` có P-frame BD-rate +0,77% theo PCHIP nhưng −2,80% theo cubic; của `temporal` là +1,71% và −2,91%. Không chọn cách nội suy theo dấu có lợi. Các đường đo và cả hai kết quả phải được giữ lại.

**Ưu tiên tiếp theo:** dùng spatial + ΔQP làm mốc phát triển trên validation; kiểm tra cơ chế temporal bằng các ablation MC/GRU/EMA/loss riêng rẽ, đồng thời đo temporal residual và thống kê encoder. Kiểm tra độ phù hợp của tiêu chí chọn checkpoint (val loss so với rate/AP trên MOT17-09). Sau khi chốt thiết kế bằng validation, thực hiện một protocol mới với GOP 8/16/32 và bộ dữ liệu video thứ hai. Không tối ưu lặp lại trên test MOT17-11/13 để rồi dùng chính test đó làm bằng chứng cuối.


## So sánh trực tiếp giữa các nhánh

Mỗi so sánh được tính lại từ hai đường rate/AP của cùng seed. Không lấy hiệu hai BD-rate so với x265. Dấu âm nghĩa là nhánh ở cột Test tốt hơn nhánh Anchor.

| So sánh | Anchor → Test | Tổng — Faster R-CNN AP50–95 | Tổng — YOLO AP50–95 | P-frame — YOLO AP50–95 |
|---|---|---:|---:|---:|
| `temporal_without_qg` | `spatial` → `temporal` | +1.71 ± 0.56 | +2.00 ± 0.48 | +9.10 ± 4.27 |
| `temporal_with_qg` | `spatial_qp` → `temporal_qp` | +5.45 ± 7.13 | +5.29 ± 4.10 | +6.96 ± 6.36 |
| `qg_without_temporal` | `spatial` → `spatial_qp` | -3.84 ± 2.46 | -1.47 ± 1.75 | +1.69 ± 0.47 |
| `qg_with_temporal` | `temporal` → `temporal_qp` | -0.86 ± 3.87 | +1.30 ± 2.59 | -0.75 ± 6.16 |
| `real_codec_vs_proxy` | `temporal_qp_proxy` → `temporal_qp` | -0.41 ± 0.53 | +1.72 ± 3.91 | +2.55 ± 4.59 |
| `full_vs_scope_adapted` | `scope_paper_adapted` → `temporal_qp` | -7.79 ± 0.46 | -2.48 ± 2.49 | +5.53 ± 2.68 |
| `spatial_qp_vs_scope_adapted` | `scope_paper_adapted` → `spatial_qp` | -12.09 ± 5.68 | -7.12 ± 5.46 | -0.92 ± 6.55 |

## Độ nhạy với nội suy cubic

Bảng dưới dùng bitrate tổng và AP50–95. Cubic là phân tích độ nhạy, không thay thế PCHIP đã chốt trong protocol.

| Nhánh | FRCNN PCHIP | FRCNN cubic | YOLO PCHIP | YOLO cubic | Lệch lớn nhất giữa hai cách, mọi seed (điểm %) |
|---|---:|---:|---:|---:|---:|
| `spatial` | -11.16 ± 3.38 | -14.36 ± 2.82 | -10.35 ± 4.35 | -11.14 ± 2.51 | 3.75 |
| `spatial_qp` | -14.88 ± 5.51 | -17.01 ± 5.40 | -11.78 ± 5.67 | -14.46 ± 5.31 | 2.99 |
| `temporal` | -9.75 ± 3.43 | -12.30 ± 2.08 | -8.57 ± 4.70 | -11.62 ± 5.34 | 4.14 |
| `temporal_qp` | -10.82 ± 0.94 | -14.24 ± 1.67 | -7.45 ± 2.54 | -9.98 ± 1.93 | 4.81 |
| `temporal_qp_proxy` | -10.31 ± 1.40 | -13.81 ± 2.08 | -8.94 ± 2.26 | -13.29 ± 0.80 | 5.93 |
| `scope_paper_adapted` | -3.11 ± 1.47 | -7.02 ± 0.72 | -5.11 ± 1.36 | -7.64 ± 1.72 | 4.84 |

## Theo từng seed: AP50–95

| Nhánh | Seed | Tổng FRCNN | Tổng YOLO | P-frame FRCNN | P-frame YOLO |
|---|---:|---:|---:|---:|---:|
| `spatial` | 2026 | -7.7882 | -6.8171 | -7.1673 | -5.7694 |
| `spatial` | 2027 | -11.1201 | -9.0228 | -4.6714 | -1.4910 |
| `spatial` | 2028 | -14.5573 | -15.2080 | -12.2873 | -13.1968 |
| `spatial_qp` | 2026 | -8.8484 | -6.3380 | -7.0778 | -3.5524 |
| `spatial_qp` | 2027 | -16.1077 | -11.3491 | -7.9730 | -0.7965 |
| `spatial_qp` | 2028 | -19.6698 | -17.6614 | -14.8026 | -11.7971 |
| `temporal` | 2026 | -6.4907 | -4.4498 | +4.2430 | +7.4470 |
| `temporal` | 2027 | -9.4197 | -7.5819 | +2.3531 | +5.2838 |
| `temporal` | 2028 | -13.3339 | -13.6877 | -7.1283 | -7.6132 |
| `temporal_qp` | 2026 | -11.3163 | -5.2419 | -2.6562 | +6.5597 |
| `temporal_qp` | 2027 | -9.7411 | -6.8777 | -5.9604 | -1.8532 |
| `temporal_qp` | 2028 | -11.4071 | -10.2242 | -4.1313 | -2.4020 |
| `temporal_qp_proxy` | 2026 | -10.8168 | -10.8597 | -1.0546 | -1.0711 |
| `temporal_qp_proxy` | 2027 | -8.7301 | -6.4459 | -6.0086 | -2.7768 |
| `temporal_qp_proxy` | 2028 | -11.3934 | -9.5091 | -3.9184 | -1.1202 |
| `scope_paper_adapted` | 2026 | -3.2990 | -3.7052 | -1.1311 | -1.6990 |
| `scope_paper_adapted` | 2027 | -1.5472 | -6.4185 | +0.0703 | -6.6899 |
| `scope_paper_adapted` | 2028 | -4.4713 | -5.2042 | -4.3051 | -5.3135 |

## Bảng bổ sung: AP50

| Nhánh | Tổng FRCNN | Tổng YOLO | P-frame FRCNN | P-frame YOLO |
|---|---:|---:|---:|---:|
| `spatial` | -15.12 ± 2.60 | -15.19 ± 4.58 | -13.55 ± 5.44 | -13.60 ± 8.16 |
| `spatial_qp` | -22.04 ± 4.28 | -17.96 ± 5.28 | -20.21 ± 2.44 | -14.33 ± 5.90 |
| `temporal` | -11.60 ± 6.43 | -10.65 ± 3.77 | -2.87 ± 10.97 | -1.47 ± 7.27 |
| `temporal_qp` | -18.03 ± 1.55 | -15.00 ± 1.35 | -14.63 ± 2.93 | -10.30 ± 2.02 |
| `temporal_qp_proxy` | -15.50 ± 1.26 | -13.52 ± 1.61 | -11.13 ± 5.25 | -8.32 ± 4.36 |
| `scope_paper_adapted` | -4.76 ± 2.84 | -4.52 ± 1.92 | -4.06 ± 3.70 | -3.73 ± 3.46 |

## Điểm đo theo CRF

Các số dưới là trung bình theo ba seed để tiện đọc; BD-rate phía trên được tính riêng từng seed rồi mới tổng hợp. Không tính BD-rate từ các điểm trung bình trong bảng này.

| Nhánh | CRF | Total bpp | I bpp | P bpp | FRCNN AP50–95 (%) | YOLO AP50–95 (%) |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_x265` | 22 | 0.6606 | 1.4567 | 0.3952 | 33.0820 | 29.3710 |
| `anchor_x265` | 27 | 0.3352 | 0.8652 | 0.1586 | 31.6990 | 28.1910 |
| `anchor_x265` | 32 | 0.1662 | 0.4849 | 0.0600 | 28.6790 | 25.6370 |
| `anchor_x265` | 37 | 0.0795 | 0.2538 | 0.0214 | 22.4160 | 20.2550 |
| `spatial` | 22 | 0.5745 | 1.2478 | 0.3501 | 33.1087 | 29.3893 |
| `spatial` | 27 | 0.2998 | 0.7583 | 0.1470 | 31.6683 | 28.2557 |
| `spatial` | 32 | 0.1526 | 0.4367 | 0.0579 | 28.8517 | 25.7237 |
| `spatial` | 37 | 0.0752 | 0.2363 | 0.0214 | 23.3873 | 20.7637 |
| `spatial_qp` | 22 | 0.5076 | 1.1208 | 0.3032 | 32.8427 | 29.1157 |
| `spatial_qp` | 27 | 0.2662 | 0.6804 | 0.1281 | 31.4393 | 27.8480 |
| `spatial_qp` | 32 | 0.1363 | 0.3915 | 0.0513 | 28.4467 | 25.2920 |
| `spatial_qp` | 37 | 0.0678 | 0.2125 | 0.0195 | 22.7473 | 20.0837 |
| `temporal` | 22 | 0.5622 | 1.1956 | 0.3511 | 32.9737 | 29.3503 |
| `temporal` | 27 | 0.2934 | 0.7255 | 0.1494 | 31.5670 | 28.0690 |
| `temporal` | 32 | 0.1490 | 0.4176 | 0.0594 | 28.6220 | 25.5277 |
| `temporal` | 37 | 0.0731 | 0.2260 | 0.0222 | 23.0087 | 20.4097 |
| `temporal_qp` | 22 | 0.5269 | 1.1605 | 0.3156 | 33.0690 | 29.2523 |
| `temporal_qp` | 27 | 0.2741 | 0.6998 | 0.1322 | 31.4333 | 27.9133 |
| `temporal_qp` | 32 | 0.1389 | 0.3984 | 0.0524 | 28.2260 | 25.0470 |
| `temporal_qp` | 37 | 0.0681 | 0.2129 | 0.0198 | 22.1293 | 19.5613 |
| `temporal_qp_proxy` | 22 | 0.5197 | 1.1478 | 0.3104 | 32.9280 | 29.1937 |
| `temporal_qp_proxy` | 27 | 0.2708 | 0.6927 | 0.1302 | 31.3227 | 27.7613 |
| `temporal_qp_proxy` | 32 | 0.1375 | 0.3951 | 0.0516 | 28.1670 | 25.1857 |
| `temporal_qp_proxy` | 37 | 0.0676 | 0.2121 | 0.0195 | 22.1460 | 19.6877 |
| `scope_paper_adapted` | 22 | 0.5790 | 1.2973 | 0.3396 | 32.9623 | 29.3290 |
| `scope_paper_adapted` | 27 | 0.3004 | 0.7815 | 0.1401 | 31.4047 | 28.0560 |
| `scope_paper_adapted` | 32 | 0.1525 | 0.4463 | 0.0546 | 28.3167 | 25.4487 |
| `scope_paper_adapted` | 37 | 0.0750 | 0.2393 | 0.0203 | 22.5783 | 20.2653 |

## Phạm vi bằng chứng

Kết quả có ba lần train độc lập cho mỗi nhánh, cùng dữ liệu/codec/source và detector YOLO chưa tham gia training. Độ lệch chuẩn mô tả biến động theo seed; 1.648 frame không phải 1.648 lần lặp độc lập. Với n=3 và hai sequence, không kết luận ý nghĩa thống kê hay khái quát sang dữ liệu/codec khác.

`scope_paper_adapted` là reference được viết lại, dùng MOT17/x265 và objective chung của dự án. Không gọi đây là tái lập chính thức SCOPE và không đối chiếu trực tiếp phần trăm từ bài gốc.

Gói temporal gồm MC, GRU, EMA và các loss temporal. Bộ thí nghiệm hiện tại chưa tách đóng góp từng phần. Bitrate P-frame kém hơn không đủ để suy ra cơ chế nguyên nhân ở motion compensation hoặc inter prediction.

## Kiểm tra và khả năng tái lập

Mỗi job được tính AP toàn bộ test bằng `scripts/merge_v3.py --allow-partial`. Bước gộp dùng lại các AP đó, đồng thời kiểm tra lại SHA raw/metadata/checkpoint, cùng source/config/data, frame/GT/ignore giữa 18 job và tính lại bitrate total/I/P từ packet bytes. Tất cả BD-rate và so sánh giữa nhánh đều được tính lại từ các điểm đã xác nhận. Không gộp trọng số, không lấy trung bình AP từng clip, không chọn seed đẹp.

- Scientific source revision: `92d778c356331f5fb3d8f9e66f7388b33821a1ad`.
- Benchmark SHA-256: `a8bd1fb6b25221fdf7fe71bf977a979f5593fc73359193f8f3818e1572187c15`.
- Full benchmark: `outputs_v3/evaluation_20261117/benchmark_v3.json`.
- Machine-readable assessment, all curves/seed diagnostics: `outputs_v3/evaluation_20261117/BD_RATE_ASSESSMENT.json`.
- Analysis helpers: `.runtime/merge_verified_v3_cache.py`, `.runtime/summarize_final_bdrate.py`.
- Independent slow reproduction: run `scripts/merge_v3.py --input-dir outputs_v3/evaluation_20261117/results --output <new-file.json>`.

## Notebook evaluation đã hoàn tất

| Job | Notebook |
|---|---|
| `spatial_seed2026` | [nguyenhoanglan1232/tscope-v3-eval-1117-spatial-seed2026](https://www.kaggle.com/code/nguyenhoanglan1232/tscope-v3-eval-1117-spatial-seed2026) |
| `spatial_seed2027` | [vtk269/tscope-v3-eval-1117-spatial-seed2027](https://www.kaggle.com/code/vtk269/tscope-v3-eval-1117-spatial-seed2027) |
| `spatial_seed2028` | [wagur124705/tscope-v3-eval-1117-spatial-seed2028](https://www.kaggle.com/code/wagur124705/tscope-v3-eval-1117-spatial-seed2028) |
| `spatial_qp_seed2026` | [hieusunday0412/tscope-v3-eval-1117-spatial-qp-seed2026](https://www.kaggle.com/code/hieusunday0412/tscope-v3-eval-1117-spatial-qp-seed2026) |
| `spatial_qp_seed2027` | [thuhuyntrn/tscope-v3-eval-1117-spatial-qp-seed2027](https://www.kaggle.com/code/thuhuyntrn/tscope-v3-eval-1117-spatial-qp-seed2027) |
| `spatial_qp_seed2028` | [tranthihongdieu/tscope-v3-eval-1117-spatial-qp-seed2028](https://www.kaggle.com/code/tranthihongdieu/tscope-v3-eval-1117-spatial-qp-seed2028) |
| `temporal_seed2026` | [linhowi05/tscope-v3-eval-1117-temporal-seed2026](https://www.kaggle.com/code/linhowi05/tscope-v3-eval-1117-temporal-seed2026) |
| `temporal_seed2027` | [hoangminhhuy123/tscope-v3-eval-1117-temporal-seed2027](https://www.kaggle.com/code/hoangminhhuy123/tscope-v3-eval-1117-temporal-seed2027) |
| `temporal_seed2028` | [thuha1205/tscope-v3-eval-1117-temporal-seed2028](https://www.kaggle.com/code/thuha1205/tscope-v3-eval-1117-temporal-seed2028) |
| `temporal_qp_seed2026` | [shungg05/tscope-v3-eval-1117-temporal-qp-seed2026](https://www.kaggle.com/code/shungg05/tscope-v3-eval-1117-temporal-qp-seed2026) |
| `temporal_qp_seed2027` | [thuhuyntrn/tscope-eval-tqp2027-move-run-f0b8dc5f](https://www.kaggle.com/code/thuhuyntrn/tscope-eval-tqp2027-move-run-f0b8dc5f) |
| `temporal_qp_seed2028` | [huolgggnuyen/tscope-v3-eval-1117-temporal-qp-seed2028](https://www.kaggle.com/code/huolgggnuyen/tscope-v3-eval-1117-temporal-qp-seed2028) |
| `temporal_qp_proxy_seed2026` | [baooo25r/tscope-v3-eval-1117-temporal-qp-proxy-seed2026](https://www.kaggle.com/code/baooo25r/tscope-v3-eval-1117-temporal-qp-proxy-seed2026) |
| `temporal_qp_proxy_seed2027` | [trmnguyn111/tscope-v3-eval-1117-temporal-qp-proxy-seed2027](https://www.kaggle.com/code/trmnguyn111/tscope-v3-eval-1117-temporal-qp-proxy-seed2027) |
| `temporal_qp_proxy_seed2028` | [line344/tscope-v3-eval-1117-temporal-qp-proxy-seed2028](https://www.kaggle.com/code/line344/tscope-v3-eval-1117-temporal-qp-proxy-seed2028) |
| `scope_paper_adapted_seed2026` | [nguoingoaihanhlang/tscope-v3-eval-1117-scope-paper-adapted-seed2026](https://www.kaggle.com/code/nguoingoaihanhlang/tscope-v3-eval-1117-scope-paper-adapted-seed2026) |
| `scope_paper_adapted_seed2027` | [nguyntrnhongan/tscope-v3-eval-1117-scope-paper-adapted-seed2027](https://www.kaggle.com/code/nguyntrnhongan/tscope-v3-eval-1117-scope-paper-adapted-seed2027) |
| `scope_paper_adapted_seed2028` | [electus225/tscope-v3-eval-1117-scope-paper-adapted-seed2028](https://www.kaggle.com/code/electus225/tscope-v3-eval-1117-scope-paper-adapted-seed2028) |

Bản JSON công bố kèm báo cáo: [V3_BDRATE_RESULTS.json](V3_BDRATE_RESULTS.json).
