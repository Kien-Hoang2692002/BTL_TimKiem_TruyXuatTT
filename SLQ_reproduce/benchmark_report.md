# BÁO CÁO TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐỒNG NHẤT (BF16)
Dự án WandB: `slq-benchmark-final` | Phần cứng: 4x NVIDIA A100 SXM4 80GB (bfloat16, Batch 256, 5 epochs)

## 1. Mốc chuẩn SLQ Gốc & Reproduce
| Phương pháp / Cấu hình                    | Params | Time  | VRAM | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| ----------------------------------------- | ------ | ----- | ---- | ---------- | ---------- | ---------- | ------------ | ------------ |
| SLQ Gốc (Bài báo công bố)                 | 19.7K  | —     | —    | 61.20%     | 84.90%     | 48.50%     | 84.20%       | 74.80%       |
| exp1_baseline_l1_mean (Reproduce L1 Mean) | 19.7K  | 17.5m | 9.8G | 60.10%     | 85.12%     | 47.75%     | 84.40%       | 73.06%       |

## 2. Ablation Study 1: Vị trí tiêm Query (Injection Depth)
*Cố định: 20 Queries, Layer 24 Mean Pooling (19.7K params)*

| Phương pháp / Cấu hình      | Params | Time  | VRAM | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| --------------------------- | ------ | ----- | ---- | ---------- | ---------- | ---------- | ------------ | ------------ |
| Tầng 1 (SLQ gốc - 0% depth) | 19.7K  | 17.5m | 9.8G | 60.10%     | 85.12%     | 47.75%     | 84.40%       | 73.06%       |
| Tầng 8 (33.3% depth)        | 19.7K  | 14.0m | 9.0G | 63.28%     | 86.40%     | 49.93%     | 86.10%       | 75.78%       |
| Tầng 12 (50.0% depth)       | 19.7K  | 12.0m | 8.5G | 63.40%     | 86.64%     | 50.38%     | 86.30%       | 76.32%       |
| Tầng 16 (66.7% depth)       | 19.7K  | 10.0m | 8.0G | 63.54%     | 86.52%     | 49.76%     | 86.40%       | 75.46%       |
| Tầng 20 (83.3% depth)       | 19.7K  | 8.0m  | 7.5G | 60.50%     | 84.70%     | 47.70%     | 85.60%       | 72.98%       |

## 3. Ablation Study 2: Chiến lược trích xuất tầng & Projector Head
*Cố định: Tiêm tại Tầng 1 (0% depth)*

| Phương pháp / Cấu hình                                  | Params | Time  | VRAM  | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| ------------------------------------------------------- | ------ | ----- | ----- | ---------- | ---------- | ---------- | ------------ | ------------ |
| 2 tầng cuối [23, 24] + Mean Pooling                     | 19.7K  | 17.6m | 9.9G  | 60.50%     | 84.94%     | 47.61%     | 85.30%       | 73.46%       |
| 4 tầng cuối [21, 22, 23, 24] + Mean Pooling             | 19.7K  | 17.6m | 10.2G | 60.44%     | 84.08%     | 47.14%     | 82.70%       | 72.52%       |
| 4 tầng cuối [21, 22, 23, 24] + Residual Refinement      | 3.23M  | 17.5m | 10.2G | 64.94%     | 87.78%     | 50.68%     | 86.60%       | 75.70%       |
| 4 tầng cách nhau [12, 16, 20, 24] + Residual Refinement | 3.23M  | 17.6m | 10.2G | 62.50%     | 86.92%     | 49.56%     | 85.90%       | 74.42%       |

## 4. Ablation Study 3: Phân bố không gian Query (Prefix - Suffix Ratio)
*Cố định: Tiêm tại Tầng 1, Layer 24 Mean Pooling*

| Phương pháp / Cấu hình         | Params | Time  | VRAM | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| ------------------------------ | ------ | ----- | ---- | ---------- | ---------- | ---------- | ------------ | ------------ |
| 0 Prefix + 20 Suffix (SLQ gốc) | 19.7K  | 17.5m | 9.8G | 60.10%     | 85.12%     | 47.75%     | 84.40%       | 73.06%       |
| 5 Prefix + 15 Suffix           | 17.9K  | 17.5m | 9.8G | 62.22%     | 85.52%     | 48.38%     | 85.90%       | 73.44%       |
| 10 Prefix + 10 Suffix          | 17.9K  | 17.5m | 9.8G | 60.96%     | 85.02%     | 47.70%     | 83.30%       | 72.92%       |
| 15 Prefix + 5 Suffix           | 17.9K  | 17.5m | 9.8G | 61.24%     | 85.38%     | 48.49%     | 86.00%       | 73.92%       |

## 5. Các mô hình phối hợp Đỉnh cao (Synergy Champions)
| Phương pháp / Cấu hình                                     | Params | Time | VRAM | COCO I2T@1 | COCO I2T@5 | COCO T2I@1 | Flickr I2T@1 | Flickr T2I@1 |
| ---------------------------------------------------------- | ------ | ---- | ---- | ---------- | ---------- | ---------- | ------------ | ------------ |
| exp_lqi18_last4_residual                                   | —      | —    | —    | —          | —          | —          | —            | —            |
| exp_hybrid_5pre_15suf_lqi18_last4_residual (Kỷ lục 62.66%) | —      | —    | —    | —          | —          | —          | —            | —            |
