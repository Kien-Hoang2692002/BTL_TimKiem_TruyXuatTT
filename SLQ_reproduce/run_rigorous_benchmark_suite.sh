#!/usr/bin/env bash
set -e

# ==============================================================================
# BỘ THỰC NGHIỆM ĐỒNG NHẤT CHUẨN HỌC THUẬT (RIGOROUS BENCHMARK SUITE - 12 RUNS)
# Tất cả đều chạy trên cùng cấu hình:
# - Precision: bfloat16 (chuẩn A100 Tensor Cores)
# - Hardware: 4x NVIDIA A100-SXM4-80GB (DDP torchrun)
# - Global Batch Size: 256 x 4 x 1 = 1024 (chuẩn bài báo gốc)
# - ViT Offline Caching: Bỏ qua ViT, tốc độ tối đa
# - Epochs: 5, LR: 5e-4 (Cosine Annealing + Warmup)
#
# Danh sách 12 Thí nghiệm (Chia thành 4 nhóm độc lập):
#   [Nhóm 1] Baseline SLQ Gốc:
#     1. exp1_baseline_l1_mean
#   [Nhóm 2] Cải tiến 1 - Vị trí tiêm Query (Injection Depth Ablation):
#     2. exp2a_lqi8_l24_mean
#     3. exp2b_lqi12_l24_mean
#     4. exp2c_lqi16_l24_mean
#     5. exp2d_lqi20_l24_mean
#   [Nhóm 3] Cải tiến 2 - Chiến lược trích xuất tầng & Head (Multi-layer Strategy Ablation):
#     6. exp3a_last2_mean       (2 tầng cuối [23, 24] + Mean Pooling)
#     7. exp3b_last4_mean       (4 tầng cuối [21, 22, 23, 24] + Mean Pooling)
#     8. exp3c_last4_residual   (4 tầng cuối [21, 22, 23, 24] + Residual Head)
#     9. exp3d_spaced4_residual (4 tầng cách đều [12, 16, 20, 24] + Residual Head)
#   [Nhóm 4] Cải tiến 3 - Phân bố không gian Query (Prefix-Suffix Ratio Ablation):
#     10. exp4a_split_5pre_15suf_mean  (5 Prefix + 15 Suffix, Tầng 1, Mean L24)
#     11. exp4b_split_10pre_10suf_mean (10 Prefix + 10 Suffix, Tầng 1, Mean L24)
#     12. exp4c_split_15pre_5suf_mean  (15 Prefix + 5 Suffix, Tầng 1, Mean L24)
#
# Cách chạy:
#   chmod +x run_rigorous_benchmark_suite.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_rigorous_benchmark_suite.sh /cm/archive/quangnm594/data/coco2017 coco_train2017_vit_cache.pt
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
MASTER_PORT=${MASTER_PORT:-29505}
WANDB_PROJECT=${WANDB_PROJECT:-"slq-retrieval"}

echo "=================================================================="
echo "    BỘ THỰC NGHIỆM ĐỒNG NHẤT CHUẨN HỌC THUẬT (12 RUNS BF16)"
echo "=================================================================="
echo "COCO Root            : $COCO_ROOT"
echo "Cache File           : $CACHE_FILE"
echo "Số lượng GPUs (DDP)  : $NUM_GPUS A100"
echo "Precision            : bfloat16 (Tự động)"
echo "Global Batch Size    : $BATCH_SIZE x $NUM_GPUS x $GRAD_ACCUM = $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo "Số Epochs            : $EPOCHS"
echo "Wandb Project        : $WANDB_PROJECT"
echo "=================================================================="

# Kiểm tra file cache
if [ ! -f "$CACHE_FILE" ]; then
    echo "LỖI: Chưa tìm thấy file $CACHE_FILE. Vui lòng kiểm tra lại đường dẫn!"
    exit 1
fi

# ==============================================================================
# [NHÓM 1] BASELINE REPRODUCE: SLQ NGUYÊN BẢN (TẦNG 1, MEAN POOLING L24)
# ==============================================================================
echo -e "\n>>> [1/12] NHÓM 1: BASELINE SLQ GỐC (20 Queries L1, L24 Mean Pooling)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --inject_layer 1 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp1_baseline_l1_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp1_baseline_l1_mean"

# ==============================================================================
# [NHÓM 2] CẢI TIẾN 1: VỊ TRÍ TIÊM QUERY (LATE INJECTION DEPTH ABLATION)
# Cố định: 20 Queries, Layer 24 Mean Pooling (19.7K params)
# ==============================================================================
echo -e "\n>>> [2/12] NHÓM 2 (Ablation 1): INJECT TẦNG 8 + L24 MEAN (33.3% depth)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --inject_layer 8 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp2a_lqi8_l24_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp2a_lqi8_l24_mean"

echo -e "\n>>> [3/12] NHÓM 2 (Ablation 1): INJECT TẦNG 12 + L24 MEAN (50.0% depth)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --inject_layer 12 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp2b_lqi12_l24_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp2b_lqi12_l24_mean"

echo -e "\n>>> [4/12] NHÓM 2 (Ablation 1): INJECT TẦNG 16 + L24 MEAN (66.7% depth)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --inject_layer 16 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp2c_lqi16_l24_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp2c_lqi16_l24_mean"

echo -e "\n>>> [5/12] NHÓM 2 (Ablation 1): INJECT TẦNG 20 + L24 MEAN (83.3% depth)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --inject_layer 20 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp2d_lqi20_l24_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp2d_lqi20_l24_mean"

# ==============================================================================
# [NHÓM 3] CẢI TIẾN 2: CHIẾN LƯỢC TRÍCH XUẤT TẦNG & HEAD (MULTI-LAYER STRATEGY)
# Cố định: Inject tại Tầng 1
# ==============================================================================
echo -e "\n>>> [6/12] NHÓM 3 (Ablation 2): 2 TẦNG CUỐI [23, 24] + MEAN POOLING..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --inject_layer 1 --layers 23,24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp3a_last2_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp3a_last2_mean"

echo -e "\n>>> [7/12] NHÓM 3 (Ablation 2): 4 TẦNG CUỐI [21, 22, 23, 24] + MEAN POOLING..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --inject_layer 1 --layers 21,22,23,24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp3b_last4_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp3b_last4_mean"

echo -e "\n>>> [8/12] NHÓM 3 (Ablation 2): 4 TẦNG CUỐI [21, 22, 23, 24] + RESIDUAL REFINEMENT..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --inject_layer 1 --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp3c_last4_residual.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp3c_last4_residual"

echo -e "\n>>> [9/12] NHÓM 3 (Ablation 2): 4 TẦNG CÁCH NHAU [12, 16, 20, 24] + RESIDUAL REFINEMENT..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --inject_layer 1 --layers 12,16,20,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp3d_spaced4_residual.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp3d_spaced4_residual"

# ==============================================================================
# [NHÓM 4] CẢI TIẾN 3: PHÂN BỐ KHÔNG GIAN QUERY (PREFIX-SUFFIX RATIO ABLATION)
# Cố định: Inject tại Tầng 1, Tầng 24 Mean Pooling
# ==============================================================================
echo -e "\n>>> [10/12] NHÓM 4 (Ablation 3): 5 PREFIX + 15 SUFFIX (TẦNG 1, MEAN L24)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --num_prefix_queries 5 --inject_layer 1 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp4a_split_5pre_15suf_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp4a_split_5pre_15suf_mean"

echo -e "\n>>> [11/12] NHÓM 4 (Ablation 3): 10 PREFIX + 10 SUFFIX (TẦNG 1, MEAN L24)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --num_prefix_queries 10 --inject_layer 1 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp4b_split_10pre_10suf_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp4b_split_10pre_10suf_mean"

echo -e "\n>>> [12/12] NHÓM 4 (Ablation 3): 15 PREFIX + 5 SUFFIX (TẦNG 1, MEAN L24)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --num_prefix_queries 15 --inject_layer 1 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp4c_split_15pre_5suf_mean.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp4c_split_15pre_5suf_mean"

echo -e "\n=================================================================="
echo ">>> HOÀN THÀNH TOÀN BỘ 12 RUNS THỰC NGHIỆM!"
echo ">>> Chạy 'python SLQ_reproduce/aggregate_benchmark.py' để xuất bảng tổng hợp khoa học!"
echo "=================================================================="
