#!/usr/bin/env bash
set -e

# ==============================================================================
# Script CHẠY SONG SONG TRÊN CỤM 4 GPU A100 (Parallel Multi-GPU Runner)
# Chia đều 9 thí nghiệm lên 4 GPU để chạy đồng thời:
#   - GPU 0: [0a] Last Token Zero-shot + [0b] Last Token Linear + [1] Baseline SLQ
#   - GPU 1: [2] 4 Layer cuối (Mean) + [3] 4 Layer cuối (MLP)
#   - GPU 2: [4] 4 Layer cuối (Residual) + [5] 4 Layer cách đều (Mean)
#   - GPU 3: [6] 4 Layer cách đều (MLP) + [7] 4 Layer cách đều (Residual)
#
# Toàn bộ 9 thí nghiệm hoàn thành trong khoảng ~7-10 phút!
#
# Cách dùng:
#   chmod +x run_all_experiments_parallel.sh
#   ./run_all_experiments_parallel.sh [ĐƯỜNG_DẪN_COCO2017]
# ==============================================================================

COCO_ROOT=${1:-"./data/coco2017"}
MAX_PAIRS=${MAX_PAIRS:-0}       # 0 = toàn bộ ~118k ảnh COCO (hoặc đặt 20000)
EPOCHS=${EPOCHS:-5}             # 5 epochs chuẩn bài báo
BATCH_SIZE=${BATCH_SIZE:-128}   # Micro batch size trên A100
GRAD_ACCUM=${GRAD_ACCUM:-8}     # 128 x 8 = 1024 (Batch size toàn cục như bài báo)
LR=5e-4                         # Learning rate bài báo gốc
WORKERS=4                       # 4 workers per GPU (tổng 16 workers)
CACHE_FILE=${2:-"coco_train2017_vit_cache.pt"}

echo "=================================================================="
echo "    CHẠY SONG SONG 9 THÍ NGHIỆM SLQ TRÊN CỤM 4 GPU NVIDIA A100    "
echo "=================================================================="
echo "COCO Root            : $COCO_ROOT"
echo "Cache File           : $CACHE_FILE"
echo "Dữ liệu huấn luyện   : $( [ "$MAX_PAIRS" -eq 0 ] && echo "Toàn bộ ~118k ảnh COCO (Chuẩn bài báo)" || echo "$MAX_PAIRS ảnh" )"
echo "Số Epochs            : $EPOCHS"
echo "Batch Size mỗi GPU   : $BATCH_SIZE x $GRAD_ACCUM = 1024 (Chuẩn bài báo)"
echo "Wandb Tracking       : slq-retrieval"
echo "Phân bổ GPUs         : GPU 0, 1, 2, 3 chạy song song"
echo "=================================================================="

# 1. Tự động tải dữ liệu COCO nếu chưa có
COCO_ANN_FOUND=0
for cand in "$COCO_ROOT/annotations/captions_train2017.json" "$COCO_ROOT/captions_train2017.json"; do
    if [ -f "$cand" ]; then
        COCO_ANN_FOUND=1
        break
    fi
done

if [ "$COCO_ANN_FOUND" -eq 0 ]; then
    echo -e "\n>>> Chưa tìm thấy dữ liệu COCO tại $COCO_ROOT. Tự động tải qua download_coco.sh..."
    bash download_coco.sh "$COCO_ROOT"
fi

# 2. Trích xuất ViT Feature Cache một lần duy nhất nếu chưa có (dùng GPU 0)
if [ ! -f "$CACHE_FILE" ]; then
    echo -e "\n>>> BƯỚC 0: TRÍCH XUẤT TRƯỚC VIT FEATURES TRÊN GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python precompute_vit.py \
        --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --out "$CACHE_FILE" --batch_size "$BATCH_SIZE"
fi

echo -e "\n>>> BẮT ĐẦU PHÂN PHỐI 9 THÍ NGHIỆM ĐỒNG THỜI LÊN 4 GPU..."

# ----------------- Worker GPU 0 -----------------
(
    echo "[GPU 0] Bắt đầu [0a] Last Token Zero-shot..."
    CUDA_VISIBLE_DEVICES=0 python slq_multilayer.py eval \
        --data flickr,coco \
        --num_queries 0 --layers 24 --head_type none --queries init \
        --batch_size "$BATCH_SIZE" \
        --out backbone_last_token_zeroshot.pt \
        --wandb_run_name "exp0a_backbone_last_token_zeroshot" > log_gpu0.txt 2>&1

    echo "[GPU 0] Bắt đầu [0b] Last Token Linear Head..."
    CUDA_VISIBLE_DEVICES=0 python slq_multilayer.py train \
        --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
        --num_queries 0 --layers 24 --head_type linear --lr "$LR" --epochs "$EPOCHS" \
        --out slq_backbone_last_token_linear.pt \
        --wandb_run_name "exp0b_backbone_last_token_linear" >> log_gpu0.txt 2>&1

    echo "[GPU 0] Bắt đầu [1] Baseline SLQ (Layer 24, Mean)..."
    CUDA_VISIBLE_DEVICES=0 python slq_multilayer.py train \
        --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
        --num_queries 20 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
        --out slq_baseline_q20.pt \
        --wandb_run_name "exp1_baseline_l24" >> log_gpu0.txt 2>&1

    echo "[GPU 0] Hoàn thành toàn bộ nhiệm vụ!"
) &
PID0=$!

# ----------------- Worker GPU 1 -----------------
(
    echo "[GPU 1] Bắt đầu [2] 4 Layer cuối + Mean..."
    CUDA_VISIBLE_DEVICES=1 python slq_multilayer.py train \
        --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
        --layers 21,22,23,24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
        --out slq_last4_mean_q20.pt \
        --wandb_run_name "exp2_last4_mean" > log_gpu1.txt 2>&1

    echo "[GPU 1] Bắt đầu [3] 4 Layer cuối + MLP..."
    CUDA_VISIBLE_DEVICES=1 python slq_multilayer.py train \
        --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
        --layers 21,22,23,24 --head_type mlp --lr "$LR" --epochs "$EPOCHS" \
        --out slq_last4_mlp_q20.pt \
        --wandb_run_name "exp3_last4_mlp" >> log_gpu1.txt 2>&1

    echo "[GPU 1] Hoàn thành toàn bộ nhiệm vụ!"
) &
PID1=$!

# ----------------- Worker GPU 2 -----------------
(
    echo "[GPU 2] Bắt đầu [4] 4 Layer cuối + Residual..."
    CUDA_VISIBLE_DEVICES=2 python slq_multilayer.py train \
        --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
        --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
        --out slq_last4_residual_q20.pt \
        --wandb_run_name "exp4_last4_residual" > log_gpu2.txt 2>&1

    echo "[GPU 2] Bắt đầu [5] 4 Layer cách đều + Mean..."
    CUDA_VISIBLE_DEVICES=2 python slq_multilayer.py train \
        --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
        --layers 6,12,18,24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
        --out slq_spaced_mean_q20.pt \
        --wandb_run_name "exp5_spaced_mean" >> log_gpu2.txt 2>&1

    echo "[GPU 2] Hoàn thành toàn bộ nhiệm vụ!"
) &
PID2=$!

# ----------------- Worker GPU 3 -----------------
(
    echo "[GPU 3] Bắt đầu [6] 4 Layer cách đều + MLP..."
    CUDA_VISIBLE_DEVICES=3 python slq_multilayer.py train \
        --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
        --layers 6,12,18,24 --head_type mlp --lr "$LR" --epochs "$EPOCHS" \
        --out slq_spaced_mlp_q20.pt \
        --wandb_run_name "exp6_spaced_mlp" > log_gpu3.txt 2>&1

    echo "[GPU 3] Bắt đầu [7] 4 Layer cách đều + Residual..."
    CUDA_VISIBLE_DEVICES=3 python slq_multilayer.py train \
        --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
        --layers 6,12,18,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
        --out slq_spaced_residual_q20.pt \
        --wandb_run_name "exp7_spaced_residual" >> log_gpu3.txt 2>&1

    echo "[GPU 3] Hoàn thành toàn bộ nhiệm vụ!"
) &
PID3=$!

echo "Các tiến trình đang chạy ngầm trên GPU 0 (PID $PID0), GPU 1 (PID $PID1), GPU 2 (PID $PID2), GPU 3 (PID $PID3)..."
echo "Bạn có thể theo dõi tiến độ qua lệnh: tail -f log_gpu0.txt (hoặc log_gpu1.txt, log_gpu2.txt, log_gpu3.txt)"

# Đợi tất cả 4 workers hoàn thành
wait $PID0
wait $PID1
wait $PID2
wait $PID3

echo -e "\n=================================================================="
echo ">>> TOÀN BỘ 4 GPU ĐÃ HOÀN THÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ!"
echo "=================================================================="

# Xuất bảng tổng hợp kết quả
python summarize_results.py

