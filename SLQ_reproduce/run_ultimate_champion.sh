#!/usr/bin/env bash
set -e

# ==============================================================================
# RUN VÔ ĐỊCH TỐI THƯỢNG (ULTIMATE GRAND CHAMPION RUN)
# Hợp nhất 4 đột phá lớn nhất của toàn bộ nghiên cứu:
#   1. Global Task Conditioning: 5 Prefix Queries tại Tầng 1 (định hướng tác vụ từ gốc)
#   2. Late Query Injection: Tiêm tại Tầng 12 (sau khi ngữ cảnh cấp cao đã chín muồi)
#   3. Instance-Adaptive Generator: 15 Suffix Queries sinh động qua MLP từ Last Token Tầng 11
#   4. Noise Filtering Head: 4-Layer Residual Refinement Head [21, 22, 23, 24] (khử nhiễu autoregressive)
#
# Cấu hình huấn luyện:
# - Precision: bfloat16
# - Hardware: 4x NVIDIA A100-SXM4-80GB (DDP torchrun)
# - Global Batch Size: 256 x 4 x 1 = 1024
# - Attention Backend: eager
# - Project Wandb: slq-benchmark-final
#
# Cách chạy:
#   chmod +x run_ultimate_champion.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_ultimate_champion.sh /cm/archive/quangnm594/data/coco2017 coco_train2017_vit_cache.pt
# ==============================================================================

COCO_ROOT=${1:-"./data/coco2017"}
CACHE_FILE=${2:-"coco_train2017_vit_cache.pt"}
MAX_PAIRS=${MAX_PAIRS:-0}       # 0 = toàn bộ ~118k ảnh COCO
EPOCHS=${EPOCHS:-5}             # 5 epochs chuẩn bài báo
NUM_GPUS=${NUM_GPUS:-4}         # 4 GPU A100
BATCH_SIZE=${BATCH_SIZE:-256}   # Micro batch size 256
GRAD_ACCUM=${GRAD_ACCUM:-1}     # Global Batch Size = 256 x 4 x 1 = 1024
LR=5e-4
WORKERS=6
MASTER_PORT=${MASTER_PORT:-29513}
WANDB_PROJECT=${3:-${WANDB_PROJECT:-"slq-benchmark-final"}}
ATTN_IMPL="eager"

echo "=================================================================="
echo "    KHỞI CHẠY ULTIMATE GRAND CHAMPION (ADAPTIVE MLP + RESIDUAL HEAD)"
echo "=================================================================="
echo "COCO Root            : $COCO_ROOT"
echo "Cache File           : $CACHE_FILE"
echo "Số lượng GPUs (DDP)  : $NUM_GPUS A100"
echo "Precision            : bfloat16 (Tự động)"
echo "Attention Backend    : $ATTN_IMPL"
echo "Cấu trúc Query       : 5 Prefix (Tầng 1) + 15 Adaptive Suffix (MLP Tầng 12)"
echo "Head Projector       : 4-Layer Residual Refinement [21, 22, 23, 24]"
echo "Global Batch Size    : $BATCH_SIZE x $NUM_GPUS x $GRAD_ACCUM = $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo "Số Epochs            : $EPOCHS"
echo "Wandb Project        : $WANDB_PROJECT"
echo "=================================================================="

# Kiểm tra file cache
if [ ! -f "$CACHE_FILE" ]; then
    echo "CẢNH BÁO: Chưa tìm thấy file $CACHE_FILE. Vui lòng kiểm tra lại đường dẫn."
fi

echo -e "\n>>> ĐANG CHẠY MÔ HÌNH VÔ ĐỊCH TỐI THƯỢNG (ULTIMATE GRAND CHAMPION)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --attn_implementation "$ATTN_IMPL" \
    --adaptive_mode mlp \
    --inject_layer 12 \
    --num_queries 20 \
    --num_prefix_queries 5 \
    --layers 21,22,23,24 \
    --head_type residual \
    --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp_ultimate_champion_lqi12_5pre_adaptive_mlp15suf_last4_residual.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp_ultimate_champion_lqi12_5pre_adaptive_mlp15suf_last4_residual"

echo -e "\n=================================================================="
echo "    HOÀN TẤT ULTIMATE GRAND CHAMPION RUN! ĐANG TỰ ĐỘNG CẬP NHẬT BÁO CÁO..."
echo "=================================================================="
python aggregate_benchmark.py --project "$WANDB_PROJECT"

