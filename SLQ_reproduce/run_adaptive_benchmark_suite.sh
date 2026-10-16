#!/usr/bin/env bash
set -e

# ==============================================================================
# BỘ THỰC NGHIỆM ADAPTIVE QUERY BENCHMARK (INSTANCE-ADAPTIVE QUERIES AT LAYER 12)
# Khảo sát 2 cơ chế thích ứng theo đầu vào từ Last Token Tầng 11:
#   1. Ý tưởng 1: Direct MLP Generator (exp_adaptive_mlp_lqi12_mean)
#   2. Ý tưởng 2a: DeepSeek Sparse MoE Query Pool (Pool 30, 5 Shared + 15 Routed)
#   3. Ý tưởng 2b: DeepSeek Sparse MoE Query Pool (Pool 60, 5 Shared + 15 Routed)
#   4. Đỉnh cao: DeepSeek MoE Query Pool + 4-Layer Residual Refinement Head
#
# Cấu hình huấn luyện:
# - Precision: bfloat16
# - Hardware: 4x NVIDIA A100-SXM4-80GB (DDP torchrun)
# - Global Batch Size: 256 x 4 x 1 = 1024
# - Attention Backend: eager
# - Project Wandb: slq-benchmark-final
#
# Cách chạy:
#   chmod +x run_adaptive_benchmark_suite.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_adaptive_benchmark_suite.sh /cm/archive/quangnm594/data/coco2017 coco_train2017_vit_cache.pt
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
MASTER_PORT=${MASTER_PORT:-29511}
WANDB_PROJECT=${3:-${WANDB_PROJECT:-"slq-benchmark-final"}}
ATTN_IMPL="eager"

echo "=================================================================="
echo "    BỘ THỰC NGHIỆM INSTANCE-ADAPTIVE QUERIES TẠI TẦNG 12"
echo "=================================================================="
echo "COCO Root            : $COCO_ROOT"
echo "Cache File           : $CACHE_FILE"
echo "Số lượng GPUs (DDP)  : $NUM_GPUS A100"
echo "Precision            : bfloat16 (Tự động)"
echo "Attention Backend    : $ATTN_IMPL"
echo "Global Batch Size    : $BATCH_SIZE x $NUM_GPUS x $GRAD_ACCUM = $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo "Số Epochs            : $EPOCHS"
echo "Wandb Project        : $WANDB_PROJECT"
echo "=================================================================="

# Kiểm tra file cache
if [ ! -f "$CACHE_FILE" ]; then
    echo "CẢNH BÁO: Chưa tìm thấy file $CACHE_FILE. Vui lòng kiểm tra lại đường dẫn."
fi

# ==============================================================================
# [RUN 1/4] Ý TƯỞNG 1: MLP GENERATOR (SINH 20 QUERIES TỪ LAST TOKEN L11)
# Đối chứng trực tiếp với exp2b_lqi12_l24_mean (63.40%)
# ==============================================================================
echo -e "\n>>> [1/4] Ý TƯỞNG 1: MLP QUERY GENERATOR TẠI L12 (20 QUERIES, L24 MEAN)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --attn_implementation "$ATTN_IMPL" \
    --adaptive_mode mlp \
    --inject_layer 12 \
    --num_queries 20 \
    --layers 24 \
    --head_type mean \
    --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp_adaptive_mlp_lqi12_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp_adaptive_mlp_lqi12_mean"

# ==============================================================================
# [RUN 2/4] Ý TƯỞNG 2A: DEEPSEEK SPARSE MOE QUERY POOL (POOL 30 = 2X)
# 5 Shared Queries + 15 Routed Queries chọn từ Pool 30 queries với Loss-Free Balance
# ==============================================================================
echo -e "\n>>> [2/4] Ý TƯỞNG 2A: DEEPSEEK MOE QUERY POOL 30 (5 SHARED + 15 ROUTED, L24 MEAN)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --attn_implementation "$ATTN_IMPL" \
    --adaptive_mode moe_pool \
    --moe_pool_size 30 \
    --moe_shared_queries 5 \
    --moe_routed_queries 15 \
    --loss_free_gamma 0.01 \
    --inject_layer 12 \
    --layers 24 \
    --head_type mean \
    --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp_adaptive_moe30_lqi12_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp_adaptive_moe30_lqi12_mean"

# ==============================================================================
# [RUN 3/4] Ý TƯỞNG 2B: DEEPSEEK SPARSE MOE QUERY POOL (POOL 60 = 4X)
# Khảo sát mở rộng dung lượng Pool từ 30 lên 60 queries
# ==============================================================================
echo -e "\n>>> [3/4] Ý TƯỞNG 2B: DEEPSEEK MOE QUERY POOL 60 (5 SHARED + 15 ROUTED, L24 MEAN)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --attn_implementation "$ATTN_IMPL" \
    --adaptive_mode moe_pool \
    --moe_pool_size 60 \
    --moe_shared_queries 5 \
    --moe_routed_queries 15 \
    --loss_free_gamma 0.01 \
    --inject_layer 12 \
    --layers 24 \
    --head_type mean \
    --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp_adaptive_moe60_lqi12_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp_adaptive_moe60_lqi12_mean"

# ==============================================================================
# [RUN 4/4] ĐỈNH CAO: DEEPSEEK MOE QUERY POOL + 4-LAYER RESIDUAL REFINEMENT HEAD
# Kết hợp cả cơ chế thích ứng đầu vào (MoE Pool) và khử nhiễu đầu ra (Residual Refinement)
# ==============================================================================
echo -e "\n>>> [4/4] ĐỈNH CAO: DEEPSEEK MOE POOL 30 + 4-LAYER RESIDUAL REFINEMENT HEAD..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --attn_implementation "$ATTN_IMPL" \
    --adaptive_mode moe_pool \
    --moe_pool_size 30 \
    --moe_shared_queries 5 \
    --moe_routed_queries 15 \
    --loss_free_gamma 0.01 \
    --inject_layer 12 \
    --layers 21,22,23,24 \
    --head_type residual \
    --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp_adaptive_moe30_lqi12_last4_residual.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp_adaptive_moe30_lqi12_last4_residual"

echo -e "\n=================================================================="
echo "    HOÀN TẤT BỘ THỰC NGHIỆM ADAPTIVE! ĐANG TỰ ĐỘNG TỔNG HỢP BÁO CÁO..."
echo "=================================================================="
python aggregate_benchmark.py --project "$WANDB_PROJECT"

