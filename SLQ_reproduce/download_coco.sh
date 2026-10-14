#!/usr/bin/env bash
set -e

# ==============================================================================
# Script tự động tải và giải nén tập dữ liệu COCO 2017
# Bao gồm:
#   - annotations_trainval2017.zip (~241 MB) -> captions_train2017.json
#   - train2017.zip (~18 GB) -> ~118k ảnh train
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
echo "          TỰ ĐỘNG TẢI DỮ LIỆU HUẤN LUYỆN MS COCO 2017             "
echo "=================================================================="
echo "Thư mục lưu trữ: $TARGET_DIR"
echo "=================================================================="

# Kiểm tra công cụ tải
DOWNLOADER="wget"
if command -v aria2c >/dev/null 2>&1; then
    DOWNLOADER="aria2c"
    echo "Phát hiện aria2c (tải đa luồng siêu nhanh)!"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
else
    echo "Lỗi: Không tìm thấy wget, curl hoặc aria2c trên hệ thống."
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

# 1. Tải và giải nén Annotations (chứa captions_train2017.json)
cd "$TARGET_DIR"
if [ ! -f "annotations/captions_train2017.json" ]; then
    download_file "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "annotations_trainval2017.zip"
    echo ">>> Đang giải nén annotations_trainval2017.zip..."
    unzip -q -n annotations_trainval2017.zip
    echo "Đã giải nén xong annotations!"
else
    echo "Annotations đã tồn tại tại $TARGET_DIR/annotations/captions_train2017.json."
fi

# 2. Tải và giải nén Train Images (18 GB, ~118k ảnh)
if [ ! -d "train2017" ] || [ $(ls -1 train2017 2>/dev/null | wc -l) -lt 5000 ]; then
    download_file "http://images.cocodataset.org/zips/train2017.zip" "train2017.zip"
    echo ">>> Đang giải nén train2017.zip (khoảng 1-2 phút)..."
    unzip -q -n train2017.zip
    echo "Đã giải nén xong train2017!"
else
    echo "Thư mục ảnh train2017 đã tồn tại tại $TARGET_DIR/train2017."
fi

echo -e "\n=================================================================="
echo ">>> HOÀN TẤT CHUẨN BỊ DỮ LIỆU COCO 2017 TẠI: $TARGET_DIR"
echo "=================================================================="
