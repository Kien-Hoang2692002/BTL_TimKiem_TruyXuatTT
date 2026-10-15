#!/usr/bin/env bash
set -e

# ==============================================================================
# SCRIPT SMOKE TEST NHANH (30 GIÂY) TRÊN CỤM 4 GPU A100
# Mục đích: Kiểm tra toàn diện toàn bộ pipeline trong ~30-45s:
# 1. Khởi tạo DDP 4 GPUs (NCCL).
# 2. Cơ chế bfloat16 & ViT Cache.
# 3. Kiến trúc phức tạp nhất: Hybrid (5 Prefix L1 + 15 Suffix L18 + Residual 4 Tầng).
# 4. Huấn luyện 1 step (max_pairs 512, 1 epoch).
# 5. Lưu checkpoint.
# 6. Đánh giá song song trên cả 4 GPUs (all_gather) + tính Recall.
# ==============================================================================

COCO_ROOT=${1:-"./data/coco2017"}
CACHE_FILE=${2:-"coco_train2017_vit_cache.pt"}
ATTN_IMPL=${3:-${ATTN_IMPL:-"eager"}}
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29599}

echo "=================================================================="
echo "    BẮT ĐẦU SMOKE TEST NHANH (DỰ KIẾN ~30 - 45 GIÂY)"
echo "=================================================================="
echo "COCO Root         : $COCO_ROOT"
echo "Cache File        : $CACHE_FILE"
echo "Attention Backend : $ATTN_IMPL"
echo "Số GPUs           : $NUM_GPUS"
echo "=================================================================="

# Kiểm tra file cache
if [ ! -f "$CACHE_FILE" ]; then
    echo "LỖI: Chưa tìm thấy file $CACHE_FILE. Vui lòng kiểm tra lại đường dẫn!"
    exit 1
fi

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" \
    --cache_file "$CACHE_FILE" \
    --max_pairs 512 \
    --epochs 1 \
    --batch_size 128 \
    --grad_accum_steps 1 \
    --attn_implementation "$ATTN_IMPL" \
    --num_queries 20 \
    --num_prefix_queries 5 \
    --inject_layer 18 \
    --layers 21,22,23,24 \
    --head_type residual \
    --data flickr \
    --no_wandb \
    --out smoke_test_temp.pt

echo -e "\n=================================================================="
echo ">>> 🎉 CHÚC MỪNG: SMOKE TEST THÀNH CÔNG 100%!"
echo ">>> Toàn bộ pipeline DDP 4x A100 (Train + Eval 4 GPUs + Recall) hoạt động hoàn hảo!"
echo "=================================================================="

# Dọn dẹp file tạm
rm -f smoke_test_temp.pt results_multilayer_smoke_test_temp.json

