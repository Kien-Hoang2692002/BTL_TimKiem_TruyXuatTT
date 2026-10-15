#!/usr/bin/env bash
set -e

# ==============================================================================
# SCRIPT THỰC NGHIỆM MỞ RỘNG:
# 1. Khảo sát độ sâu Late Injection: Tầng 12 & Tầng 16 (+ Layer 24 Mean Pooling)
#    (Để so sánh trực tiếp với Tầng 1 (56.16%) và Tầng 18 (57.42%))
# 2. Kiến trúc lai Hybrid Dual-Strike: 
#    5 Prefix tokens ở Tầng 1 + 15 Suffix tokens ở Tầng 18 + 4 Tầng cuối Residual
#
# Cách chạy trên cụm 4 GPU A100:
#   chmod +x run_ablation_and_hybrid_experiments.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_ablation_and_hybrid_experiments.sh /cm/archive/quangnm594/data/coco2017 coco_train2017_vit_cache.pt
# ==============================================================================

COCO_ROOT=${1:-"./data/coco2017"}
CACHE_FILE=${2:-"coco_train2017_vit_cache.pt"}
MAX_PAIRS=${MAX_PAIRS:-0}       # 0 = toàn bộ ~118k ảnh COCO
EPOCHS=${EPOCHS:-5}             # 5 epochs chuẩn bài báo
NUM_GPUS=${NUM_GPUS:-4}         # 4 GPU A100
BATCH_SIZE=${BATCH_SIZE:-256}   # Micro batch size 256 bão hòa Tensor Cores A100
GRAD_ACCUM=${GRAD_ACCUM:-1}     # Global Batch Size = 256 x 4 x 1 = 1024
LR=5e-4
WORKERS=6
MASTER_PORT=${MASTER_PORT:-29502}

echo "=================================================================="
echo "    THỰC NGHIỆM MỞ RỘNG: LATE INJECTION (L12, L16) & HYBRID 5+15"
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
    echo "LỖI: Chưa tìm thấy file $CACHE_FILE. Vui lòng kiểm tra lại đường dẫn!"
    exit 1
fi

# ------------------------------------------------------------------------------
# [1/3] Late Injection Tầng 12 + Layer 24 Mean Pooling (50% độ sâu LLM)
# ------------------------------------------------------------------------------
echo -e "\n>>> [1/3] THÍ NGHIỆM 1: CHÈN 20 QUERIES TẦNG 12 + LAYER 24 (MEAN POOLING)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --inject_layer 12 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_lqi12_l24_mean.pt \
    --wandb_run_name "exp_lqi12_l24_mean"

# ------------------------------------------------------------------------------
# [2/3] Late Injection Tầng 16 + Layer 24 Mean Pooling (66.7% độ sâu LLM)
# ------------------------------------------------------------------------------
echo -e "\n>>> [2/3] THÍ NGHIỆM 2: CHÈN 20 QUERIES TẦNG 16 + LAYER 24 (MEAN POOLING)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --inject_layer 16 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_lqi16_l24_mean.pt \
    --wandb_run_name "exp_lqi16_l24_mean"

# ------------------------------------------------------------------------------
# [3/3] Hybrid Dual-Strike: 5 Prefix ở Tầng 1 + 15 Suffix ở Tầng 18 + Residual Head
# ------------------------------------------------------------------------------
echo -e "\n>>> [3/3] THÍ NGHIỆM 3: HYBRID (5 PREFIX L1 + 15 SUFFIX L18) + 4 LAYER CUỐI RESIDUAL..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --num_prefix_queries 5 --inject_layer 18 \
    --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_hybrid_5pre_15suf_l18_residual.pt \
    --wandb_run_name "exp_hybrid_5pre_15suf_lqi18_last4_residual"

echo -e "\n=================================================================="
echo ">>> HOÀN THÀNH TẤT CẢ 3 THÍ NGHIỆM MỞ RỘNG!"
echo "=================================================================="

