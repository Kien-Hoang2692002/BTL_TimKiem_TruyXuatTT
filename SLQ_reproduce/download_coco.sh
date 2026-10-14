#!/usr/bin/env bash
set -e

# ==============================================================================
# Script tự động tải tập dữ liệu COCO 2017 qua Kaggle CLI (Tốc độ cao nhất)
#
# Lệnh sử dụng:
#   kaggle datasets download -d awsaf49/coco-2017-dataset -p [THƯ_MỤC_ĐÍCH] --unzip
#
# Cách dùng:
#   chmod +x download_coco.sh
#   ./download_coco.sh [THƯ_MỤC_ĐÍCH]
# Mặc định: ./data/coco2017
# ==============================================================================

DEST_DIR=${1:-"./data/coco2017"}
mkdir -p "$DEST_DIR"
TARGET_DIR=$(cd "$DEST_DIR" && pwd)

echo "=================================================================="
echo "      TẢI DỮ LIỆU MS COCO 2017 QUA KAGGLE CLI (GOOGLE CDN)        "
echo "=================================================================="
echo "Thư mục đích: $TARGET_DIR"
echo "Bộ dữ liệu  : awsaf49/coco-2017-dataset"
echo "=================================================================="

# 1. Kiểm tra xem dữ liệu đã tồn tại chưa
ANN_EXISTS=0
for cand in "$TARGET_DIR/annotations/captions_train2017.json" "$TARGET_DIR/captions_train2017.json" "$TARGET_DIR/coco2017/annotations/captions_train2017.json"; do
    if [ -f "$cand" ]; then
        ANN_EXISTS=1
        echo "Đã tìm thấy annotations tại: $cand"
        break
    fi
done

IMG_EXISTS=0
for cand in "$TARGET_DIR/train2017" "$TARGET_DIR/coco2017/train2017"; do
    if [ -d "$cand" ] && [ $(ls -1 "$cand" 2>/dev/null | head -n 100 | wc -l) -ge 50 ]; then
        IMG_EXISTS=1
        echo "Đã tìm thấy thư mục ảnh train2017 tại: $cand"
        break
    fi
done

if [ "$ANN_EXISTS" -eq 1 ] && [ "$IMG_EXISTS" -eq 1 ]; then
    echo -e "\n>>> Dữ liệu COCO 2017 đã tồn tại đầy đủ tại $TARGET_DIR. Bỏ qua tải."
    exit 0
fi

# 2. ƯU TIÊN 1: Dùng Kaggle CLI
USE_KAGGLE=0
if command -v kaggle >/dev/null 2>&1 || python3 -c "import kaggle" 2>/dev/null; then
    # Kiểm tra token Kaggle
    if [ -f "$HOME/.kaggle/kaggle.json" ] || [ -n "$KAGGLE_USERNAME" -a -n "$KAGGLE_KEY" ]; then
        USE_KAGGLE=1
    else
        echo ">>> Phát hiện Kaggle CLI nhưng chưa có API Token (~/.kaggle/kaggle.json hoặc KAGGLE_USERNAME / KAGGLE_KEY)."
    fi
fi

if [ "$USE_KAGGLE" -eq 1 ]; then
    echo -e "\n>>> ĐANG TẢI COCO 2017 QUA KAGGLE CLI (awsaf49/coco-2017-dataset)..."
    echo ">>> Băng thông Google Cloud CDN, tốc độ cực nhanh..."
    cd "$TARGET_DIR"
    kaggle datasets download -d awsaf49/coco-2017-dataset -p "$TARGET_DIR" --unzip
    echo -e "\n>>> Đã tải và giải nén xong qua Kaggle CLI!"
    exit 0
fi

# Nếu chưa có Kaggle API token, thông báo hướng dẫn nhanh
echo -e "\n>>> LƯU Ý: Để tải bằng Kaggle CLI nhanh nhất (100MB/s):"
echo "    1. Đăng nhập Kaggle -> Settings -> API -> Create New Token (tải file kaggle.json)"
echo "    2. Đặt file vào ~/.kaggle/kaggle.json (chmod 600 ~/.kaggle/kaggle.json)"
echo "    Hoặc gán biến môi trường: export KAGGLE_USERNAME=xxx KAGGLE_KEY=yyy"
echo "------------------------------------------------------------------"
echo ">>> Đang chuyển sang phương thức dự phòng (Download trực tiếp qua aria2c/wget)..."

# 3. PHƯƠNG THỨC DỰ PHÒNG: Tải trực tiếp từ cocodataset.org bằng aria2c hoặc wget
DOWNLOADER="wget"
if command -v aria2c >/dev/null 2>&1; then
    DOWNLOADER="aria2c"
    echo "Phát hiện aria2c (tải đa luồng 16 kết nối)!"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
else
    echo "Lỗi: Không tìm thấy công cụ tải (kaggle, aria2c, wget, curl)."
    exit 1
fi

download_file() {
    local url=$1
    local out=$2
    if [ -f "$out" ]; then
        echo "File $out đã tồn tại, bỏ qua tải."
        return 0
    fi
    echo -e "\n>>> Đang tải: $url -> $out"
    if [ "$DOWNLOADER" = "aria2c" ]; then
        aria2c -x 16 -s 16 -k 1M -c -o "$out" "$url"
    elif [ "$DOWNLOADER" = "wget" ]; then
        wget -c -O "$out" "$url"
    else
        curl -C - -L -o "$out" "$url"
    fi
}

cd "$TARGET_DIR"
# Tải và giải nén Annotations
if [ ! -f "annotations/captions_train2017.json" ]; then
    download_file "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "annotations_trainval2017.zip"
    echo ">>> Đang giải nén annotations..."
    unzip -q -n annotations_trainval2017.zip
fi

# Tải và giải nén train2017
if [ ! -d "train2017" ] || [ $(ls -1 train2017 2>/dev/null | wc -l) -lt 5000 ]; then
    download_file "http://images.cocodataset.org/zips/train2017.zip" "train2017.zip"
    echo ">>> Đang giải nén train2017.zip..."
    unzip -q -n train2017.zip
fi

echo -e "\n=================================================================="
echo ">>> HOÀN TẤT CHUẨN BỊ DỮ LIỆU COCO 2017 TẠI: $TARGET_DIR"
echo "=================================================================="
