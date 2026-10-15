#!/usr/bin/env bash
set -e

# ==============================================================================
# BỘ 3 THỰC NGHIỆM CUỐI CÙNG HOÀN THIỆN ĐỒ ÁN / PAPER (COMPLETION SUITE)
# Gồm:
#   1. exp0a_backbone_last_token_zeroshot: Mốc sàn Zero-shot (Không train, chỉ eval)
#   2. exp0b_backbone_last_token_linear:   Mốc đối chứng truyền thống (Table 5 paper)
#   3. exp_grand_champion_lqi12_5pre_15suf_last4_residual: Mô hình Vô địch Tổng hợp
#
# Cấu hình phần cứng & huấn luyện:
# - Precision: bfloat16 (chuẩn A100 Tensor Cores)
# - Hardware: 4x NVIDIA A100-SXM4-80GB (DDP torchrun)
# - Global Batch Size: 256 x 4 x 1 = 1024 (chuẩn bài báo gốc)
# - Attention Backend: Eager Attention (Chuẩn mực an toàn, tránh lỗi padding)
# - Wandb Project: slq-benchmark-final
#
# Cách chạy:
#   chmod +x run_final_completion_suite.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_final_completion_suite.sh /cm/archive/quangnm594/data/coco2017 coco_train2017_vit_cache.pt
# ==============================================================================

COCO_ROOT=${1:-"./data/coco2017"}
CACHE_FILE=${2:-"coco_train2017_vit_cache.pt"}
MAX_PAIRS=${MAX_PAIRS:-0}       # 0 = toàn bộ ~118k ảnh COCO
EPOCHS=${EPOCHS:-5}             # 5 epochs chuẩn bài báo
NUM_GPUS=${NUM_GPUS:-4}         # 4 GPU A100
BATCH_SIZE=${BATCH_SIZE:-256}   # Micro batch size 256 trên A100
GRAD_ACCUM=${GRAD_ACCUM:-1}     # Global Batch Size = 256 x 4 x 1 = 1024
LR=5e-4
WORKERS=6
MASTER_PORT=${MASTER_PORT:-29509}
WANDB_PROJECT=${3:-${WANDB_PROJECT:-"slq-benchmark-final"}}
ATTN_IMPL="eager"

echo "=================================================================="
echo "    BỘ 3 THỰC NGHIỆM CUỐI CÙNG HOÀN THIỆN (FINAL COMPLETION SUITE)"
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
    echo "CẢNH BÁO: Chưa tìm thấy file $CACHE_FILE. Nếu chạy train, script sẽ tự động tạo cache."
fi

# ==============================================================================
# [1/3] MỐC SÀN ĐỐI CHỨNG: ZERO-SHOT BACKBONE LAST TOKEN (KHÔNG TRAIN, CHỈ EVAL)
# Thời gian chạy: ~1.5 - 2 phút trên 4 GPU A100
# ==============================================================================
echo -e "\n>>> [1/3] MỐC SÀN: BACKBONE LAST TOKEN (ZERO-SHOT EVAL TRÊN FLICKR & COCO)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py eval \
    --data flickr,coco \
    --num_queries 0 \
    --layers 24 \
    --head_type none \
    --queries init \
    --batch_size 128 \
    --attn_implementation "$ATTN_IMPL" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp0a_backbone_last_token_zeroshot"

# ==============================================================================
# [2/3] MỐC ĐỐI CHỨNG TRUYỀN THỐNG: BACKBONE LAST TOKEN + LINEAR HEAD (TABLE 5 PAPER)
# Đóng băng backbone, chỉ học Linear Projector từ Last Token (4.19M params, 5 Epochs)
# Thời gian chạy: ~15 phút
# ==============================================================================
echo -e "\n>>> [2/3] MỐC ĐỐI CHỨNG TRUYỀN THỐNG: BACKBONE LAST TOKEN + LINEAR HEAD (TABLE 5 PAPER)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --attn_implementation "$ATTN_IMPL" \
    --num_queries 0 --layers 24 --head_type linear --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp0b_backbone_last_token_linear.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp0b_backbone_last_token_linear"

# ==============================================================================
# [3/3] CẤU HÌNH VÔ ĐỊCH TỔNG HỢP (GRAND SYNERGY CHAMPION)
# Kết hợp bộ ba phát hiện xuất sắc nhất:
#   - 5 Prefix Queries (Tầng 1): Định hướng tác vụ toàn cục (Global steering bias)
#   - 15 Suffix Queries tiêm tại Tầng 12 (50% depth): Thu hoạch ngữ nghĩa mức cao
#   - 4 Tầng cuối [21, 22, 23, 24] + Residual Refinement Head: Khử nhiễu autoregressive
# Thời gian chạy: ~12 phút
# ==============================================================================
echo -e "\n>>> [3/3] MÔ HÌNH VÔ ĐỊCH: LQI12 + 5 PRE / 15 SUF + LAST 4 RESIDUAL REFINEMENT..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --attn_implementation "$ATTN_IMPL" \
    --num_queries 20 \
    --num_prefix_queries 5 \
    --inject_layer 12 \
    --layers 21,22,23,24 \
    --head_type residual \
    --lr "$LR" --epochs "$EPOCHS" \
    --out slq_exp_grand_champion_lqi12_5pre_15suf_last4_residual.pt \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "exp_grand_champion_lqi12_5pre_15suf_last4_residual"

echo -e "\n=================================================================="
echo "    HOÀN TẤT TOÀN BỘ 3 THỰC NGHIỆM! ĐANG TỰ ĐỘNG TỔNG HỢP BÁO CÁO..."
echo "=================================================================="
python aggregate_benchmark.py --project "$WANDB_PROJECT"
