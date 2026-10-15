#!/usr/bin/env bash
set -e

# ==============================================================================
# BỘ THÍ NGHIỆM LATE QUERY INJECTION & SEQUENCE SPLIT (PREFIX-SUFFIX) TRÊN 4 GPU A100
#
# Thí nghiệm 1: Chèn 20 prompt ở Tầng 18 + Layer 24 (Mean Pooling)
# Thí nghiệm 2: Chèn 20 prompt ở Tầng 18 + 4 Layer cuối (21,22,23,24) + Residual Refinement
# Thí nghiệm 3: Sequence Split (10 prompt đầu chuỗi + 10 prompt cuối chuỗi, chỉ pool 10 token cuối) 
#               kết hợp 4 Layer cuối (21,22,23,24) + Residual Refinement
# Thí nghiệm 4: Sequence Split (10 prompt đầu chuỗi + 10 prompt cuối chuỗi, chỉ pool 10 token cuối) 
#               + Layer 24 Mean Pooling (Đối chiếu 1:1 trực tiếp với SLQ gốc)
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
echo "    BỘ THÍ NGHIỆM LATE & SEQUENCE SPLIT INJECTION TRÊN 4 GPU A100"
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
# [1/4] Chèn Tầng 18 + Mean Pooling (Layer 24) [Đối đầu trực tiếp SLQ gốc bài báo]
# ------------------------------------------------------------------------------
echo -e "\n>>> [1/4] THÍ NGHIỆM 1: CHÈN 20 QUERIES TẦNG 18 + LAYER 24 (MEAN POOLING)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --inject_layer 18 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_lqi18_l24_mean.pt \
    --wandb_run_name "exp_lqi18_l24_mean"

# ------------------------------------------------------------------------------
# [2/4] Chèn Tầng 18 + 4 Layer cuối (21,22,23,24) + Residual Refinement
# ------------------------------------------------------------------------------
echo -e "\n>>> [2/4] THÍ NGHIỆM 2: CHÈN 20 QUERIES TẦNG 18 + 4 LAYER CUỐI (21,22,23,24) + RESIDUAL REFINEMENT..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --inject_layer 18 --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_lqi18_last4_residual.pt \
    --wandb_run_name "exp_lqi18_last4_residual"

# ------------------------------------------------------------------------------
# [3/4] Sequence Split: 10 đầu chuỗi + 10 cuối chuỗi, chỉ pool 10 cuối + 4 Layer cuối + Residual
# ------------------------------------------------------------------------------
echo -e "\n>>> [3/4] THÍ NGHIỆM 3: SEQUENCE SPLIT (10 PREFIX + 10 SUFFIX, POOL 10 CUỐI) + 4 LAYER CUỐI + RESIDUAL..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --query_split seq_split --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_seq_split10_10_last4_residual.pt \
    --wandb_run_name "exp_seq_split10_10_last4_residual"

# ------------------------------------------------------------------------------
# [4/4] Sequence Split: 10 đầu chuỗi + 10 cuối chuỗi, chỉ pool 10 cuối + Layer 24 Mean Pooling
# ------------------------------------------------------------------------------
echo -e "\n>>> [4/4] THÍ NGHIỆM 4: SEQUENCE SPLIT (10 PREFIX + 10 SUFFIX, POOL 10 CUỐI) + LAYER 24 MEAN POOLING..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --query_split seq_split --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_seq_split10_10_l24_mean.pt \
    --wandb_run_name "exp_seq_split10_10_l24_mean"

echo -e "\n=================================================================="
echo ">>> HOÀN THÀNH TẤT CẢ CÁC THÍ NGHIỆM!"
echo "=================================================================="
