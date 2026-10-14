# Hướng dẫn Chạy Huấn luyện Tối ưu SLQ trên Server NVIDIA A100

Tài liệu này hướng dẫn cách chạy phiên bản SLQ cải tiến trên GPU **NVIDIA A100 (40GB / 80GB)** để đạt hiệu năng và tốc độ tối đa.

---

## 1. Tổng quan các cải tiến đã tích hợp

| Cải tiến | Mô tả & Lợi ích |
| :--- | :--- |
| **Native bfloat16 + TF32** | Tận dụng phần cứng Tensor Cores của A100, tránh tràn số/underflow, tăng tốc tính toán ma trận. |
| **PyTorch SDPA / FlashAttention** | Sử dụng `attn_implementation="sdpa"` của PyTorch 2.x, tối ưu hóa bộ nhớ và tốc độ cho Qwen2 LLM. |
| **Offline ViT Feature Caching** | ViT & Projector bị đóng băng 100%. Trích xuất trước visual tokens `[256, 896]` một lần duy nhất vào file `.pt`. Trong quá trình train, bỏ qua 100% ViT và tiền xử lý ảnh PIL. |
| **Optimized DataLoader** | Kích hoạt `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`, `num_workers=8`. |
| **Prompt Embedding Caching** | Bộ khung nhúng ảnh `<IMG_CONTEXT>` được cache sẵn theo batch size, loại bỏ overhead gọi tokenizer lặp lại mỗi step. |
| **Residual Refinement Head** | Kiến trúc $h_{\text{final}} = h_{\text{last}} + \alpha \cdot \text{MLP}([h_6, h_{12}, h_{18}, h_{24}])$ với Zero-Init $\alpha = 0.0$. |
| **Spaced Layers `[6, 12, 18, 24]`** | Khai thác thông tin ngữ nghĩa đa tầng dọc toàn bộ 24 layer của Qwen2-0.5B. |

### So sánh tốc độ huấn luyện (20.000 cặp ảnh - caption):
- **Kaggle T4 (16GB):** ~150 phút / epoch
- **A100 (Online ViT):** ~12 - 15 phút / epoch
- **A100 (Cached ViT):** **~15 - 25 giây / epoch!** (Tốc độ bước đạt ~15 - 20 steps/s)

---

## 2. Chuẩn bị môi trường trên Server A100

Cài đặt các gói thư viện cần thiết:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers timm einops pillow datasets pandas pyarrow
```

Kiểm tra GPU:
```bash
nvidia-smi
```

---

## 3. Chạy Huấn luyện & Đánh giá

### Cách 1: Chạy tự động 1 lệnh bằng script bash
Chỉ cần chỉ định đường dẫn thư mục COCO 2017:
```bash
cd SLQ_reproduce
./train_a100.sh /path/to/coco2017
```
> *Lưu ý:* Lần đầu tiên chạy, script sẽ tự động trích xuất trước ViT features vào file `coco_train2017_vit_20k.pt` trong ~1-2 phút. Sau đó bước huấn luyện chỉ mất khoảng 20 giây! Các lần chạy tiếp theo sẽ dùng lại file cache này ngay lập tức.

---

### Cách 2: Chạy trực tiếp lệnh Python

#### Bước 2.1: Huấn luyện với Residual Refinement Head + Spaced Layers `[6, 12, 18, 24]`
```bash
python slq_multilayer.py train \
    --coco_root /path/to/coco2017 \
    --max_pairs 20000 \
    --cache_file coco_train2017_vit_20k.pt \
    --batch_size 64 \
    --workers 8 \
    --layers 6,12,18,24 \
    --head_type residual \
    --lr 5e-4 \
    --epochs 1 \
    --out slq_residual_spaced_q20.pt
```

#### Bước 2.2: Đánh giá trên Flickr30K và COCO-5K
```bash
python slq_multilayer.py eval \
    --data flickr,coco \
    --queries slq_residual_spaced_q20.pt \
    --layers 6,12,18,24 \
    --head_type residual \
    --batch_size 64
```

---

## 4. Thử nghiệm các kiến trúc Head khác (Tùy chọn)

Sau khi đã có file `coco_train2017_vit_20k.pt`, bạn có thể huấn luyện thử nghiệm bất kỳ phương án nào chỉ trong vòng **vài chục giây**:

### A. Query Cross-Attention Head (`cross_attn`)
```bash
python slq_multilayer.py train \
    --coco_root /path/to/coco2017 \
    --cache_file coco_train2017_vit_20k.pt \
    --layers 6,12,18,24 \
    --head_type cross_attn \
    --out slq_crossattn_spaced_q20.pt
```

### B. Multi-Layer MLP Head (`mlp`)
```bash
python slq_multilayer.py train \
    --coco_root /path/to/coco2017 \
    --cache_file coco_train2017_vit_20k.pt \
    --layers 6,12,18,24 \
    --head_type mlp \
    --out slq_mlp_spaced_q20.pt
```

### C. Trọng số học được qua Softmax (`weighted`)
```bash
python slq_multilayer.py train \
    --coco_root /path/to/coco2017 \
    --cache_file coco_train2017_vit_20k.pt \
    --layers 6,12,18,24 \
    --head_type weighted \
    --out slq_weighted_spaced_q20.pt
```

---

## 5. Theo dõi Huấn luyện & Đánh giá trên Wandb (Weights & Biases)

Hệ thống đã được tích hợp logging tự động lên Wandb:
- **Tự động kích hoạt**: Nếu trong file `.env` hoặc biến môi trường có `WANDB_API_KEY`, script sẽ tự động tạo run và đẩy thông số lên Wandb.
- **Các metrics được ghi nhận trực quan theo thời gian thực**:
  - `train/loss`: Loss hàm Contrastive InfoNCE qua từng step.
  - `train/tau`: Nhiệt độ logit scale $\tau$ học được.
  - `train/lr`: Learning rate schedule (Warmup + Cosine Decay).
  - `train/alpha`: Trọng số $\alpha$ của Residual Refinement Head qua các step.
  - `eval/flickr_i2t_r1/r5/r10` và `eval/flickr_t2i_r1/r5/r10`: Kết quả Recall trên Flickr30K.
  - `eval/coco_i2t_r1/r5/r10` và `eval/coco_t2i_r1/r5/r10`: Kết quả Recall trên COCO-5K.

**Các cờ Wandb tùy biến**:
- `--wandb_project <tên_project>` (mặc định: `slq-retrieval`)
- `--wandb_run_name <tên_run>` (nếu không đặt sẽ tự sinh theo tên mô hình và config)
- `--no_wandb`: Tắt Wandb nếu muốn chạy offline hoàn toàn.

---

## 6. Tùy chỉnh tham số nâng cao trên A100
- **`--batch_size`**: Trên A100 80GB, có thể nâng lên `128` (hoặc `256` khi dùng cache). Batch size càng lớn thì số lượng in-batch negatives trong contrastive learning càng phong phú, giúp mô hình phân biệt tốt hơn.
- **`--max_pairs`**: Mặc định là `20000`. Nếu muốn huấn luyện trên toàn bộ ~118k ảnh của COCO 2017, hãy đặt `--max_pairs 0`.
- **`--epochs`**: Do thời gian train với cache chỉ mất ~20 giây/epoch, bạn có thể dễ dàng thử train `3` hoặc `5` epochs với cosine learning rate decay.


