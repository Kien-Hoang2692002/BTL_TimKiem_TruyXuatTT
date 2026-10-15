#!/usr/bin/env bash
set -e

# ==============================================================================
# NGHIÊN CỨU MỞ RỘNG: LATE QUERY INJECTION (LQI) / MID-LAYER INJECTION
#
# Giả thuyết khoa học:
#   Thay vì chèn 20 Query tokens ngay tại Layer 1 (gây nhiễu biểu diễn ban đầu của
#   backbone LLM), ta để các token hình ảnh và văn bản tự tương tác thuần túy đến
#   tầng sâu (Layer 18, 20 hoặc 23), khi các đặc trưng đã chín muồi mới chèn Query
#   tokens đóng vai trò như một Attentive Reader / Resampler để đóng gói biểu diễn.
#
# Chuỗi thí nghiệm:
#   [LQI-1] Chèn tại Layer 20 + 4 Layer cuối (21,22,23,24) + Residual Refinement
#   [LQI-2] Chèn tại Layer 20 + 1 Layer cuối 24 (So sánh trực tiếp với SLQ gốc)
#   [LQI-3] Chèn tại Layer 23 + 1 Layer cuối 24 (Kiểm chứng chèn cực muộn)
#   [LQI-4] Chèn tại Layer 18 + 4 Layer cuối (21,22,23,24) + Residual Refinement
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
echo "    NGHIÊN CỨU MỞ RỘNG: LATE QUERY INJECTION (LQI) TRÊN 4 GPU A100"
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
# [LQI-1] Chèn Layer 20 + 4 Layer cuối + Residual Refinement
# ------------------------------------------------------------------------------
echo -e "\n>>> [LQI-1] CHÈN LAYER 20 + 4 LAYER CUỐI (21,22,23,24) + RESIDUAL REFINEMENT..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --inject_layer 20 --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_lqi20_last4_residual.pt \
    --wandb_run_name "exp_lqi20_last4_residual"

# ------------------------------------------------------------------------------
# [LQI-2] Chèn Layer 20 + 1 Layer cuối 24 (So sánh đối đầu với SLQ gốc bài báo)
# ------------------------------------------------------------------------------
echo -e "\n>>> [LQI-2] CHÈN LAYER 20 + 1 LAYER 24 (MEAN POOLING, SO SÁNH SLQ GỐC)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --inject_layer 20 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_lqi20_l24_mean.pt \
    --wandb_run_name "exp_lqi20_l24_mean"

# ------------------------------------------------------------------------------
# [LQI-3] Chèn Layer 23 + 1 Layer cuối 24 (Kiểm chứng chèn cực muộn)
# ------------------------------------------------------------------------------
echo -e "\n>>> [LQI-3] CHÈN LAYER 23 + 1 LAYER 24 (KIỂM CHỨNG CHÈN CỰC MUỘN)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --inject_layer 23 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_lqi23_l24_mean.pt \
    --wandb_run_name "exp_lqi23_l24_mean"

# ------------------------------------------------------------------------------
# [LQI-4] Chèn Layer 18 + 4 Layer cuối (21,22,23,24) + Residual Refinement
# ------------------------------------------------------------------------------
echo -e "\n>>> [LQI-4] CHÈN LAYER 18 + 4 LAYER CUỐI (21,22,23,24) + RESIDUAL REFINEMENT..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --inject_layer 18 --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_lqi18_last4_residual.pt \
    --wandb_run_name "exp_lqi18_last4_residual"

echo -e "\n=================================================================="
echo ">>> HOÀN THÀNH TẤT CẢ CÁC THÍ NGHIỆM LATE QUERY INJECTION (LQI)!"
echo "=================================================================="
