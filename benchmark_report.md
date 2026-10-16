# BÁO CÁO KHOA HỌC: CẢI TIẾN TRUY VẤN TIỀM ẨN THÍCH ỨNG ĐA TẦNG CHO MLLM RETRIEVAL
Dự án WandB: `slq-benchmark-final` | Phần cứng: 4x NVIDIA A100 SXM4 80GB (bfloat16, Batch 256, 5 epochs)

> [!NOTE]
> **Lưu ý về Tốc độ (Wall-clock Time):** Các thực nghiệm ở Mục 1 đến 4 chạy trên GPU sạch (100% idle). Các thực nghiệm Adaptive và Ultimate ở Mục 5, 6 chạy trong phiên sau bị tranh chấp tài nguyên với tiến trình GPU khác trên server, nên thời gian thực thi bị kéo dài hơn. Vì vậy, chỉ nên so sánh thời gian tương đối trong cùng một phiên đo đạc.

## PHẦN I: BÁO CÁO THỰC NGHIỆM CHÍNH THỨC (OFFICIAL BENCHMARK REPORT)

### 1. Tái lập Phương pháp Gốc & Các Mốc Đối chứng Cơ sở
*Mục tiêu: Thiết lập mốc sàn tự nhiên của LLM tiền huấn luyện và mốc đối chứng với phương pháp học Linear Head truyền thống.*

| Phương pháp / Cấu hình                          | Params | Time  | VRAM | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| ----------------------------------------------- | ------ | ----- | ---- | ---------- | ---------- | ---------- | ------------ | ------------ |
| Backbone Zero-shot (Last Token)                 | —      | —     | —    | 2.04%      | 7.28%      | 1.52%      | 4.60%        | 4.02%        |
| Backbone + Linear Head (Table 5 Paper)          | 803.7K | 5.0m  | 6.8G | 37.08%     | 66.92%     | 28.03%     | 55.40%       | 47.56%       |
| SLQ Gốc (Bài báo công bố)                       | 19.7K  | —     | —    | 61.20%     | 84.90%     | 48.50%     | 84.20%       | 74.80%       |
| SLQ Tái lập (Baseline: 20 Queries L1, L24 Mean) | 19.7K  | 17.5m | 9.8G | 60.10%     | 85.12%     | 47.75%     | 84.40%       | 73.06%       |

### 2. Nghiên cứu Phân bố Không gian Query (Prefix-Steering Queries)
*Cố định: Tiêm tại Tầng 1, Layer 24 Mean Pooling. Khảo sát việc phân tách query thành Prefix (đầu chuỗi) và Suffix (đuôi chuỗi).*

| Phương pháp / Cấu hình                        | Params | Time  | VRAM | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| --------------------------------------------- | ------ | ----- | ---- | ---------- | ---------- | ---------- | ------------ | ------------ |
| 0 Prefix + 20 Suffix (SLQ gốc)                | 19.7K  | 17.5m | 9.8G | 60.10%     | 85.12%     | 47.75%     | 84.40%       | 73.06%       |
| 5 Prefix + 15 Suffix (Best Prefix Steering) ⭐ | 17.9K  | 17.5m | 9.8G | 62.22%     | 85.52%     | 48.38%     | 85.90%       | 73.44%       |
| 10 Prefix + 10 Suffix                         | 17.9K  | 17.5m | 9.8G | 60.96%     | 85.02%     | 47.70%     | 83.30%       | 72.92%       |
| 15 Prefix + 5 Suffix                          | 17.9K  | 17.5m | 9.8G | 61.24%     | 85.38%     | 48.49%     | 86.00%       | 73.92%       |

### 3. Nghiên cứu Vị trí Tiêm Query: Tiêm Trễ Tầng Giữa (Late Query Injection - LQI)
*Cố định: 20 Queries, Layer 24 Mean Pooling (19.7K params). Khảo sát độ sâu lý tưởng để bắt đầu tương tác cross-attention.*

| Phương pháp / Cấu hình                           | Params | Time  | VRAM | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| ------------------------------------------------ | ------ | ----- | ---- | ---------- | ---------- | ---------- | ------------ | ------------ |
| Tiêm Tầng 1 (0.0% Depth - SLQ gốc)               | 19.7K  | 17.5m | 9.8G | 60.10%     | 85.12%     | 47.75%     | 84.40%       | 73.06%       |
| Tiêm Tầng 8 (33.3% Depth)                        | 19.7K  | 14.0m | 9.0G | 63.28%     | 86.40%     | 49.93%     | 86.10%       | 75.78%       |
| Tiêm Tầng 12 (50.0% Depth - LQI-12 Sweet Spot) ⭐ | 19.7K  | 12.0m | 8.5G | 63.40%     | 86.64%     | 50.38%     | 86.30%       | 76.32%       |
| Tiêm Tầng 16 (66.7% Depth)                       | 19.7K  | 10.0m | 8.0G | 63.54%     | 86.52%     | 49.76%     | 86.40%       | 75.46%       |
| Tiêm Tầng 20 (83.3% Depth)                       | 19.7K  | 8.0m  | 7.5G | 60.50%     | 84.70%     | 47.70%     | 85.60%       | 72.98%       |

### 4. Nghiên cứu Trích xuất Đa tầng & Khối Tinh chỉnh Residual (Multi-Layer Residual Refinement)
*Cố định: Tiêm tại Tầng 1. Khảo sát các chiến lược trích xuất tầng sâu và kiến trúc Projector Head để lọc nhiễu autoregressive.*

| Phương pháp / Cấu hình                                | Params | Time  | VRAM  | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| ----------------------------------------------------- | ------ | ----- | ----- | ---------- | ---------- | ---------- | ------------ | ------------ |
| 2 tầng cuối [23, 24] + Mean Pooling                   | 19.7K  | 17.6m | 9.9G  | 60.50%     | 84.94%     | 47.61%     | 85.30%       | 73.46%       |
| 4 tầng cuối [21, 22, 23, 24] + Mean Pooling           | 19.7K  | 17.6m | 10.2G | 60.44%     | 84.08%     | 47.14%     | 82.70%       | 72.52%       |
| 4 tầng cách nhau [12, 16, 20, 24] + Residual Head     | 3.23M  | 17.6m | 10.2G | 62.50%     | 86.92%     | 49.56%     | 85.90%       | 74.42%       |
| 4 tầng cuối [21, 22, 23, 24] + Residual Head (MR-4) ⭐ | 3.23M  | 17.5m | 10.2G | 64.94%     | 87.78%     | 50.68%     | 86.60%       | 75.70%       |

### 5. Nghiên cứu Thích ứng hóa Query theo Đầu vào (Instance-Adaptive Query Generation)
*Cố định: Tiêm tại Tầng 12, Layer 24 Mean Pooling. Khảo sát khả năng sinh query động từ Last Token của Tầng 11.*

| Phương pháp / Cấu hình                                        | Params | Time  | VRAM | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| ------------------------------------------------------------- | ------ | ----- | ---- | ---------- | ---------- | ---------- | ------------ | ------------ |
| LQI-12 Tĩnh (Static Baseline)                                 | 19.7K  | 12.0m | 8.5G | 63.40%     | 86.64%     | 50.38%     | 86.30%       | 76.32%       |
| LQI-12 Thích ứng (Instance-Adaptive MLP Generator - IAQ-12) ⭐ | 9.67M  | 25.5m | 8.6G | 66.32%     | 88.60%     | 52.64%     | 88.30%       | 76.90%       |

### 6. Bảng Tổng Hợp So Sánh Toàn Diện & Mô Hình Đề Xuất (Main Benchmark)
*Tổng hợp các mốc đối chứng, phiên bản tốt nhất của từng trục cải tiến đơn lẻ, và Mô hình Đề xuất Hoàn thiện: **AdaSLQ**.*

| Phương pháp / Cấu hình                                             | Params | Time  | VRAM  | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| ------------------------------------------------------------------ | ------ | ----- | ----- | ---------- | ---------- | ---------- | ------------ | ------------ |
| Backbone + Linear Head (Table 5 Paper)                             | 803.7K | 5.0m  | 6.8G  | 37.08%     | 66.92%     | 28.03%     | 55.40%       | 47.56%       |
| SLQ Gốc (Bài báo công bố)                                          | 19.7K  | —     | —     | 61.20%     | 84.90%     | 48.50%     | 84.20%       | 74.80%       |
| SLQ Tái lập (Baseline)                                             | 19.7K  | 17.5m | 9.8G  | 60.10%     | 85.12%     | 47.75%     | 84.40%       | 73.06%       |
| + Prefix Steering (5 Pre + 15 Suf)                                 | 17.9K  | 17.5m | 9.8G  | 62.22%     | 85.52%     | 48.38%     | 85.90%       | 73.44%       |
| + Late Query Injection (LQI-12)                                    | 19.7K  | 12.0m | 8.5G  | 63.40%     | 86.64%     | 50.38%     | 86.30%       | 76.32%       |
| + Multi-Layer Residual Refinement (MR-4)                           | 3.23M  | 17.5m | 10.2G | 64.94%     | 87.78%     | 50.68%     | 86.60%       | 75.70%       |
| + Instance-Adaptive Query Generator (IAQ-12)                       | 9.67M  | 25.5m | 8.6G  | 66.32%     | 88.60%     | 52.64%     | 88.30%       | 76.90%       |
| AdaSLQ (Mô hình Đề xuất Hoàn thiện: 5-Pre + Adaptive-Suf + MR-4) 👑 | 10.59M | 37.1m | 10.3G | 69.64%     | 90.30%     | 54.38%     | 91.00%       | 78.86%       |


---
## PHẦN II: PHỤ LỤC THỰC NGHIỆM KỸ THUẬT (TECHNICAL APPENDIX)

### A. Khảo sát Mô hình Phối hợp Trung gian & Ngân hàng Query MoE
*Lưu vết các thí nghiệm thăm dò kiến trúc khác (Grand V1 tĩnh và DeepSeek MoE Query Pool).*

| Phương pháp / Cấu hình                                          | Params | Time  | VRAM  | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| --------------------------------------------------------------- | ------ | ----- | ----- | ---------- | ---------- | ---------- | ------------ | ------------ |
| Grand Champion V1 (5 Pre + 15 Suf Tĩnh + Residual Head: 68.30%) | 3.23M  | 17.5m | 10.2G | 68.30%     | 89.64%     | 53.05%     | 88.90%       | 77.32%       |
| Thăm dò: DeepSeek MoE Pool 30 (5 Shared + 15 Routed)            | 271.0K | 25.6m | 8.5G  | 63.28%     | 86.50%     | 50.01%     | 87.00%       | 75.02%       |
| Thăm dò: DeepSeek MoE Pool 60 (5 Shared + 15 Routed)            | 305.6K | 25.7m | 8.5G  | 63.74%     | 86.86%     | 50.19%     | 86.70%       | 75.78%       |
