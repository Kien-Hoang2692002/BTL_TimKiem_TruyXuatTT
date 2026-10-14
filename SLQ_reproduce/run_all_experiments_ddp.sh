#!/usr/bin/env bash
set -e

# ==============================================================================
# CHIẾN LƯỢC 2: DISTRIBUTED DATA PARALLEL (DDP) TRÊN CỤM 4 GPU A100
#
# Nguyên lý hoạt động:
#   - Sử dụng `torchrun --nproc_per_node=4` để phân tán đồng thời trên 4 GPU.
#   - Mỗi GPU gánh micro-batch: 128
#   - Tổng 4 GPU trong 1 bước: 128 x 4 = 512
#   - Tích lũy gradient: 2 bước (GRAD_ACCUM=2)
#   ==> Global Batch Size = 512 x 2 = 1024 (Đúng 100% chuẩn bài báo gốc!)
#   - InfoNCE Contrastive Loss được đồng bộ âm bản qua `torch.distributed.nn.all_gather`.
#
# Chuỗi thí nghiệm:
#   [0a] Đối chứng 1: Backbone Last Token (Zero-shot)
#   [0b] Đối chứng 2: Backbone Last Token + Linear Head (DDP 4 GPUs)
#   [1]  Baseline SLQ (Layer 24, Mean Pooling, 20 queries) (DDP 4 GPUs)
#   [2]  4 Layer cuối + Mean (DDP 4 GPUs)
#   [3]  4 Layer cuối + Concat 4 layer -> MLP (DDP 4 GPUs)
#   [4]  4 Layer cuối + Residual Refinement (DDP 4 GPUs)
#   [5]  4 Layer cách đều + Mean (DDP 4 GPUs)
#   [6]  4 Layer cách đều + Concat 4 layer -> MLP (DDP 4 GPUs)
#   [7]  4 Layer cách đều + Residual Refinement (DDP 4 GPUs)
#
# Cách dùng:
#   chmod +x run_all_experiments_ddp.sh
#   ./run_all_experiments_ddp.sh [ĐƯỜNG_DẪN_COCO2017]
# ==============================================================================

COCO_ROOT=${1:-"./data/coco2017"}
MAX_PAIRS=${MAX_PAIRS:-0}       # 0 = toàn bộ ~118k ảnh COCO (hoặc đặt 20000)
EPOCHS=${EPOCHS:-5}             # 5 epochs chuẩn bài báo
NUM_GPUS=${NUM_GPUS:-4}         # Số lượng GPU A100 (mặc định 4)
BATCH_SIZE=${BATCH_SIZE:-128}   # Micro batch size mỗi GPU
GRAD_ACCUM=${GRAD_ACCUM:-2}     # 128 x 4 GPUs x 2 accum = 1024 global batch!
LR=5e-4                         # Learning rate bài báo gốc
WORKERS=4                       # Workers mỗi GPU
MASTER_PORT=${MASTER_PORT:-29500}

CACHE_FILE=${2:-"coco_train2017_vit_cache.pt"}

echo "=================================================================="
echo "    CHIẾN LƯỢC 2: HUẤN LUYỆN DDP TRÊN CỤM $NUM_GPUS GPU NVIDIA A100       "
echo "=================================================================="
echo "COCO Root            : $COCO_ROOT"
echo "Cache File           : $CACHE_FILE"
echo "Số lượng GPUs (DDP)  : $NUM_GPUS A100"
echo "Micro-batch mỗi GPU  : $BATCH_SIZE"
echo "Gradient Accumulation: $GRAD_ACCUM bước"
echo "Global Batch Size    : $BATCH_SIZE x $NUM_GPUS x $GRAD_ACCUM = $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM)) (Chuẩn bài báo: 1024)"
echo "Số Epochs            : $EPOCHS"
echo "Learning Rate        : $LR (Warmup 3% + Cosine Decay)"
echo "Wandb Tracking       : slq-retrieval"
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
    echo -e "\n>>> BƯỚC 0: TRÍCH XUẤT TRƯỚC VIT FEATURES VÀO $CACHE_FILE (GPU 0)..."
    CUDA_VISIBLE_DEVICES=0 python precompute_vit.py \
        --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --out "$CACHE_FILE" --batch_size "$BATCH_SIZE"
fi

# =================== NHÓM ĐỐI CHỨNG: BACKBONE LAST TOKEN ===================

# 0a. ĐỐI CHỨNG 1: Backbone Last Token (Zero-shot, không huấn luyện)
echo -e "\n>>> [0a] ĐỐI CHỨNG 1: BACKBONE LAST TOKEN (ZERO-SHOT)..."
python slq_multilayer.py eval \
    --data flickr,coco \
    --num_queries 0 --layers 24 --head_type none --queries init \
    --batch_size "$BATCH_SIZE" \
    --out backbone_last_token_zeroshot.pt \
    --wandb_run_name "exp0a_backbone_last_token_zeroshot"

# 0b. ĐỐI CHỨNG 2: Backbone Last Token + Linear Head (DDP 4 GPUs)
echo -e "\n>>> [0b] ĐỐI CHỨNG 2: BACKBONE LAST TOKEN + LINEAR HEAD (DDP $NUM_GPUS GPUs, Global Batch 1024)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 0 --layers 24 --head_type linear --lr "$LR" --epochs "$EPOCHS" \
    --out slq_backbone_last_token_linear.pt \
    --wandb_run_name "exp0b_backbone_last_token_linear"

# =================== SLQ BASELINE (TÁI HIỆN BÀI BÁO GỐC) ===================

# 1. BASELINE SLQ: 1 layer 24, Mean Pooling, 20 queries (DDP 4 GPUs)
echo -e "\n>>> [1/7] CHẠY BASELINE SLQ (Layer 24, Mean, DDP $NUM_GPUS GPUs)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --num_queries 20 --layers 24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_baseline_q20.pt \
    --wandb_run_name "exp1_baseline_l24"

# =================== NHÓM 1: 4 LAYER CUỐI (21, 22, 23, 24) ===================

# 2. 4 Layer cuối + Mean
echo -e "\n>>> [2/7] 4 LAYER CUỐI + MEAN POOLING (DDP $NUM_GPUS GPUs)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 21,22,23,24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_last4_mean_q20.pt \
    --wandb_run_name "exp2_last4_mean"

# 3. 4 Layer cuối + MLP
echo -e "\n>>> [3/7] 4 LAYER CUỐI + CONCAT 4 -> MLP (DDP $NUM_GPUS GPUs)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 21,22,23,24 --head_type mlp --lr "$LR" --epochs "$EPOCHS" \
    --out slq_last4_mlp_q20.pt \
    --wandb_run_name "exp3_last4_mlp"

# 4. 4 Layer cuối + Residual
echo -e "\n>>> [4/7] 4 LAYER CUỐI + RESIDUAL REFINEMENT (DDP $NUM_GPUS GPUs)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 21,22,23,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_last4_residual_q20.pt \
    --wandb_run_name "exp4_last4_residual"

# =================== NHÓM 2: 4 LAYER CÁCH ĐỀU (6, 12, 18, 24) ===================

# 5. 4 Layer cách đều + Mean
echo -e "\n>>> [5/7] 4 LAYER CÁCH ĐỀU + MEAN POOLING (DDP $NUM_GPUS GPUs)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 6,12,18,24 --head_type mean --lr "$LR" --epochs "$EPOCHS" \
    --out slq_spaced_mean_q20.pt \
    --wandb_run_name "exp5_spaced_mean"

# 6. 4 Layer cách đều + MLP
echo -e "\n>>> [6/7] 4 LAYER CÁCH ĐỀU + CONCAT 4 -> MLP (DDP $NUM_GPUS GPUs)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 6,12,18,24 --head_type mlp --lr "$LR" --epochs "$EPOCHS" \
    --out slq_spaced_mlp_q20.pt \
    --wandb_run_name "exp6_spaced_mlp"

# 7. 4 Layer cách đều + Residual
echo -e "\n>>> [7/7] 4 LAYER CÁCH ĐỀU + RESIDUAL REFINEMENT (DDP $NUM_GPUS GPUs)..."
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" slq_multilayer.py train \
    --coco_root "$COCO_ROOT" --max_pairs "$MAX_PAIRS" --cache_file "$CACHE_FILE" \
    --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" --workers "$WORKERS" \
    --layers 6,12,18,24 --head_type residual --lr "$LR" --epochs "$EPOCHS" \
    --out slq_spaced_residual_q20.pt \
    --wandb_run_name "exp7_spaced_residual"

# =================== TỔNG HỢP VÀ IN BẢNG KẾT QUẢ ===================
echo -e "\n=================================================================="
echo ">>> TẤT CẢ CÁC THÍ NGHIỆM DDP ĐÃ HOÀN TẤT! ĐANG TỔNG HỢP BẢNG KẾT QUẢ..."
echo "=================================================================="
python summarize_results.py

echo -e "\n=================================================================="
echo ">>> HOÀN THÀNH TOÀN BỘ CHIẾN LƯỢC 2 (DDP TRÊN 4 GPU A100)!"
echo "=================================================================="

