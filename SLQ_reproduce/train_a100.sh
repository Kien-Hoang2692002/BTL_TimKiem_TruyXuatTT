#!/usr/bin/env bash
set -e

# ==============================================================================
# Script tự động chạy huấn luyện & đánh giá SLQ tối ưu trên NVIDIA A100
#
# Sử dụng:
#   chmod +x train_a100.sh
#   ./train_a100.sh [ĐƯỜNG_DẪN_COCO2017]
#
# Ví dụ:
#   ./train_a100.sh /data/coco2017
# ==============================================================================

COCO_ROOT=${1:-"/data/coco2017"}
CACHE_FILE="coco_train2017_vit_20k.pt"
CHECKPOINT="slq_residual_spaced_q20.pt"
BATCH_SIZE=64
MAX_PAIRS=20000
LAYERS="6,12,18,24"
HEAD_TYPE="residual"
LR=5e-4
EPOCHS=1
WORKERS=8

echo "=========================================================="
echo "    CHẠY HUẤN LUYỆN SLQ CẢI TIẾN TRÊN NVIDIA A100        "
echo "=========================================================="
echo "COCO Root      : $COCO_ROOT"
echo "Cache File     : $CACHE_FILE"
echo "Layers         : $LAYERS (Spaced Layers)"
echo "Head Type      : $HEAD_TYPE (Residual Refinement)"
echo "Batch Size     : $BATCH_SIZE"
echo "Max Pairs      : $MAX_PAIRS"
echo "Output Weight  : $CHECKPOINT"
echo "=========================================================="

# Kiểm tra GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Bước 1: Huấn luyện với Feature Caching & SDPA
# (Nếu CACHE_FILE chưa tồn tại, script sẽ tự động trích xuất trước trong ~1-2 phút, sau đó train trong ~20s)
echo -e "\n>>> BƯỚC 1: BẮT ĐẦU HUẤN LUYỆN..."
python slq_multilayer.py train \
    --coco_root "$COCO_ROOT" \
    --max_pairs "$MAX_PAIRS" \
    --cache_file "$CACHE_FILE" \
    --cache_batch_size "$BATCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --layers "$LAYERS" \
    --head_type "$HEAD_TYPE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --attn_implementation "sdpa" \
    --out "$CHECKPOINT"

# Bước 2: Đánh giá mô hình sau khi huấn luyện trên Flickr30K và COCO-5K
echo -e "\n>>> BƯỚC 2: ĐÁNH GIÁ (EVALUATION) TRÊN FLICKR30K VÀ COCO-5K..."
python slq_multilayer.py eval \
    --data "flickr,coco" \
    --queries "$CHECKPOINT" \
    --layers "$LAYERS" \
    --head_type "$HEAD_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --attn_implementation "sdpa"

echo -e "\n=========================================================="
echo ">>> HOÀN TẤT TẤT CẢ CÁC BƯỚC THÀNH CÔNG!"
echo "=========================================================="
