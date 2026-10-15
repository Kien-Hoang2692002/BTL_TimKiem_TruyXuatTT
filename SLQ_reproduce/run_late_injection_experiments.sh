#!/usr/bin/env bash
set -e

# ==============================================================================
# BỘ 3 THÍ NGHIỆM LATE QUERY INJECTION & SPLIT QUERY INJECTION TRÊN 4 GPU A100
#
# Thí nghiệm 1: Chèn 20 prompt ở Tầng 18 + Layer 24 (Mean Pooling)
# Thí nghiệm 2: Chèn 20 prompt ở Tầng 18 + 4 Layer cuối (21,22,23,24) + Residual Refinement
# Thí nghiệm 3: Split Query (10 prompt ở Tầng 1 + 10 prompt ở Tầng 18, chỉ pool 10 token cuối) 
#               kết hợp 4 Layer cuối + Residual Refinement
#
# Cách chạy:
#   chmod +x run_late_injection_experiments.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_late_injection_experiments.sh /cm/archive/quangnm594/data/coco2017
# ==============================================================================

COCO_ROOT=${1:-"./data/coco2017"}
CACHE_FILE=${2:-"coco_train2017_vit_cache.pt"}
MAX_PAIRS=${MAX_PAIRS:-0}       # 0 = toàn bộ ~118k ảnh COCO
EPOCHS=${EPOCHS:-5}             # 5 epochs chuẩn bài báo
NUM_GPUS=${NUM_GPUS:-4}         # 4 GPU A100
BATCH_SIZE=${BATCH_SIZE:-128}   # Micro batch size mỗi GPU
GRAD_ACCUM=${GRAD_ACCUM:-2}     # Global Batch Size = 128 x 4 x 2 = 1024
LR=5e-4
WORKERS=4
MASTER_PORT=${MASTER_PORT:-29501}

echo "=================================================================="
echo "    3 THÍ NGHIỆM LATE & SPLIT QUERY INJECTION TRÊN 4 GPU A100"
echo "=================================================================="
echo "COCO Root            : $COCO_ROOT"
echo "Cache File           : $CACHE_FILE"
echo "Số lượng GPUs (DDP)  : $NUM_GPUS A100"
echo "Global Batch Size    : $BATCH_SIZE x $NUM_GPUS x $GRAD_ACCUM = $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo "Số Epochs            : $EPOCHS"
echo "Wandb Project        : slq-retrieval"
echo "=================================================================="

# Kiểm tra file cache
if [ ! -f "$CACHE_FILE" ]; then
    echo "LỖI: Chưa tìm thấy file $CACHE_FILE. Vui lòng tạo cache trước!"
    exit 1
fi

# ------------------------------------------------------------------------------
# [1/3] Chèn Tầng 18 + Mean Pooling (Layer 24) [Đối đầu trực tiếp SLQ gốc bài báo]
# ------------------------------------------------------------------------------
echo -e "\n>>> [1/3] THÍ NGHIỆM 1: CHÈN 20 QUERIES TẦNG 18 + LAYER 24 (MEAN POOLING)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --inject_layer 18 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_lqi18_l24_mean.pt \
    --wandb_run_name "exp_lqi18_l24_mean"

# ------------------------------------------------------------------------------
# [2/3] Chèn Tầng 18 + 4 Layer cuối (21,22,23,24) + Residual Refinement
# ------------------------------------------------------------------------------
echo -e "\n>>> [2/3] THÍ NGHIỆM 2: CHÈN 20 QUERIES TẦNG 18 + 4 LAYER CUỐI (21,22,23,24) + RESIDUAL REFINEMENT..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --inject_layer 18 --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_lqi18_last4_residual.pt \
    --wandb_run_name "exp_lqi18_last4_residual"

# ------------------------------------------------------------------------------
# [3/3] Split Query: 10 ở Tầng 1 + 10 ở Tầng 18, chỉ pool 10 cuối + 4 Layer cuối + Residual
# ------------------------------------------------------------------------------
echo -e "\n>>> [3/3] THÍ NGHIỆM 3: SPLIT QUERIES (10 L1 + 10 L18, POOL 10 CUỐI) + 4 LAYER CUỐI + RESIDUAL..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --query_split layer_split --inject_layer 18 --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_split_l1_l18_last4_residual.pt \
    --wandb_run_name "exp_split_l1_l18_last4_residual"

echo -e "\n=================================================================="
echo ">>> HOÀN THÀNH TẤT CẢ 3 THÍ NGHIỆM LATE & SPLIT QUERY INJECTION!"
echo "=================================================================="
