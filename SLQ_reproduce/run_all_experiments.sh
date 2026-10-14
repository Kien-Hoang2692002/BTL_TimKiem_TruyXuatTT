#!/usr/bin/env bash
set -e

# ==============================================================================
# Script tự động chạy trọn gói 7 thí nghiệm SLQ trên NVIDIA A100:
# 1. Baseline SLQ (Layer 24, Mean)
# 2. 4 Layer cuối + Mean
# 3. 4 Layer cuối + Concat 4 layer -> MLP
# 4. 4 Layer cuối + Residual (Concat 3 layer -> MLP * alpha + Layer cuối)
# 5. 4 Layer cách đều + Mean
# 6. 4 Layer cách đều + Concat 4 layer -> MLP
# 7. 4 Layer cách đều + Residual (Concat 3 layer -> MLP * alpha + Layer cuối)
#
# Sử dụng:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh [ĐƯỜNG_DẪN_COCO2017]
# ==============================================================================

COCO_ROOT=${1:-"/data/coco2017"}
MAX_PAIRS=${MAX_PAIRS:-0}       # 0 = toàn bộ ~118k ảnh COCO như bài báo gốc (hoặc đặt 20000)
EPOCHS=${EPOCHS:-5}             # 5 epochs như bài báo gốc
BATCH_SIZE=${BATCH_SIZE:-128}   # Micro batch size trên A100
GRAD_ACCUM=${GRAD_ACCUM:-8}     # 128 x 8 = 1024 (Batch size toàn cục như bài báo gốc)
LR=5e-4                         # Learning rate bài báo gốc
WORKERS=8

CACHE_FILE=${2:-"coco_train2017_vit_cache.pt"}

echo "=================================================================="
echo "    BẮT ĐẦU CHẠY 7 THÍ NGHIỆM SLQ VỚI SIÊU THAM SỐ BÀI BÁO GỐC   "
echo "=================================================================="
echo "COCO Root            : $COCO_ROOT"
echo "Cache File           : $CACHE_FILE"
echo "Dữ liệu huấn luyện   : $( [ "$MAX_PAIRS" -eq 0 ] && echo "Toàn bộ ~118k ảnh COCO (Chuẩn bài báo)" || echo "$MAX_PAIRS ảnh" )"
echo "Số Epochs            : $EPOCHS (Chuẩn bài báo: 5)"
echo "Learning Rate        : $LR (Chuẩn bài báo: 5e-4, Cosine Decay, Warmup 3%)"
echo "Batch Size hiệu dụng : $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM)) (Chuẩn bài báo: 1024)"
echo "Wandb Tracking       : slq-retrieval"
echo "=================================================================="

# 0. Trích xuất ViT Feature Cache một lần duy nhất nếu chưa có
if [ ! -f "$CACHE_FILE" ]; then
    echo -e "\n>>> BƯỚC 0: TRÍCH XUẤT TRƯỚC VIT FEATURES VÀO $CACHE_FILE..."
    python precompute_vit.py --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --out "$CACHE_FILE" --batch_size "$BATCH_SIZE"
fi

# 1. BASELINE (Tái hiện kết quả gốc: 1 layer 24, mean pooling)
echo -e "\n>>> [1/7] CHẠY BASELINE SLQ (Layer 24, Mean, 5 Epochs, Batch 1024)..."
python slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_baseline_q20.pt \
    --wandb_run_name "exp1_baseline_l24"

# =================== NHÓM 1: 4 LAYER CUỐI (21, 22, 23, 24) ===================

# 2. 4 Layer cuối + Mean
echo -e "\n>>> [2/7] 4 LAYER CUỐI + MEAN POOLING..."
python slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 21,22,23,24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_last4_mean_q20.pt \
    --wandb_run_name "exp2_last4_mean"

# 3. 4 Layer cuối + MLP
echo -e "\n>>> [3/7] 4 LAYER CUỐI + CONCAT 4 -> MLP..."
python slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 21,22,23,24 --head_type mlp --lr "$LR" --epochs "$EPOCHS" \
    --out slq_last4_mlp_q20.pt \
    --wandb_run_name "exp3_last4_mlp"

# 4. 4 Layer cuối + Residual
echo -e "\n>>> [4/7] 4 LAYER CUỐI + RESIDUAL REFINEMENT..."
python slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_last4_residual_q20.pt \
    --wandb_run_name "exp4_last4_residual"

# =================== NHÓM 2: 4 LAYER CÁCH ĐỀU (6, 12, 18, 24) ===================

# 5. 4 Layer cách đều + Mean
echo -e "\n>>> [5/7] 4 LAYER CÁCH ĐỀU + MEAN POOLING..."
python slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 6,12,18,24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_spaced_mean_q20.pt \
    --wandb_run_name "exp5_spaced_mean"

# 6. 4 Layer cách đều + MLP
echo -e "\n>>> [6/7] 4 LAYER CÁCH ĐỀU + CONCAT 4 -> MLP..."
python slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 6,12,18,24 --head_type mlp --lr "$LR" --epochs "$EPOCHS" \
    --out slq_spaced_mlp_q20.pt \
    --wandb_run_name "exp6_spaced_mlp"

# 7. 4 Layer cách đều + Residual
echo -e "\n>>> [7/7] 4 LAYER CÁCH ĐỀU + RESIDUAL REFINEMENT..."
python slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 6,12,18,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_spaced_residual_q20.pt \
    --wandb_run_name "exp7_spaced_residual"

# =================== TỔNG HỢP VÀ IN BẢNG KẾT QUẢ ===================
echo -e "\n>>> TẤT CẢ 7 THÍ NGHIỆM ĐÃ HOÀN TẤT! ĐANG TỔNG HỢP BẢNG KẾT QUẢ..."
python summarize_results.py

echo -e "\n=================================================================="
echo ">>> HOÀN THÀNH TOÀN BỘ CHUỖI THÍ NGHIỆM!"
echo "=================================================================="

