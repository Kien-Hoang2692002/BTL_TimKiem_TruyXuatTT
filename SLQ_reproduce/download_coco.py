#!/usr/bin/env python3
"""
Tải dữ liệu MS COCO 2017 siêu tốc cho huấn luyện SLQ.

Hỗ trợ 2 chế độ:
1. mode='subset' (Mặc định & Khuyên dùng):
   - Chỉ tải annotations (~241MB) và đúng 20.000 ảnh cần thiết cho SLQ (~2.8GB).
   - Tải bất đồng bộ đa luồng (aiohttp) với 30-50 kết nối đồng thời.
   - Thời gian chỉ ~2-4 phút thay vì vài tiếng, tiết kiệm 85% dung lượng ổ cứng!
2. mode='full':
   - Tải trọn bộ 18GB (118k ảnh) bằng aria2c / kaggle API / đa luồng chunked.

Cách dùng:
  python download_coco.py --dest ./data/coco2017 --mode subset --num_images 20000
"""

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import time
import zipfile
from urllib.request import urlretrieve

try:
    import aiohttp
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm
except ImportError:
    print("Vui lòng cài đặt: pip install aiohttp tqdm")
    sys.exit(1)


ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/train2017"


def download_annotations(dest_dir: str):
    """Tải và giải nén annotations_trainval2017.zip"""
    ann_dir = os.path.join(dest_dir, "annotations")
    cap_file = os.path.join(ann_dir, "captions_train2017.json")
    if os.path.exists(cap_file):
        print(f"[1/2] Annotations đã tồn tại: {cap_file}")
        return cap_file

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "annotations_trainval2017.zip")

    if not os.path.exists(zip_path):
        print(f"[1/2] Đang tải annotations (~241MB) từ {ANNOTATIONS_URL}...")
        pbar = tqdm(unit="B", unit_scale=True, desc="Tải annotations")

        def hook(block_num, block_size, total_size):
            if pbar.total is None and total_size > 0:
                pbar.total = total_size
            pbar.update(block_size)

        urlretrieve(ANNOTATIONS_URL, zip_path, reporthook=hook)
        pbar.close()

    print(">>> Đang giải nén captions_train2017.json...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # Chỉ giải nén file annotations cần thiết để tiết kiệm thời gian & bộ nhớ
        for member in z.namelist():
            if "captions_train2017.json" in member:
                z.extract(member, dest_dir)
                break
    print(f" Đã sẵn sàng: {cap_file}")
    return cap_file


def get_target_filenames(ann_path: str, max_pairs: int = 20000, seed: int = 42):
    """Lấy danh sách file ảnh cần tải khớp chính xác 100% với logic load_coco_train_pairs trong slq.py"""
    print(f"Đang đọc {ann_path} và chọn {max_pairs} ảnh (seed={seed})...")
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2file = {im["id"]: im["file_name"] for im in data["images"]}
    annotations = data["annotations"]
    random.Random(seed).shuffle(annotations)

    seen = set()
    target_files = []
    for a in annotations:
        img_id = a["image_id"]
        if img_id in seen:
            continue
        seen.add(img_id)
        if img_id in id2file:
            target_files.append(id2file[img_id])
        if max_pairs > 0 and len(target_files) >= max_pairs:
            break

    print(f" Đã chọn {len(target_files):,} ảnh đại diện.")
    return target_files


async def download_image(session, semaphore, filename, out_dir, pbar, retries=3):
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        pbar.update(1)
        return True

    url = f"{IMAGE_BASE_URL}/{filename}"
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        with open(out_path, "wb") as f:
                            f.write(content)
                        pbar.update(1)
                        return True
            except Exception:
                await asyncio.sleep(0.5 * (attempt + 1))
        pbar.update(1)
        return False


async def download_images_async(filenames, out_dir, concurrency=40):
    os.makedirs(out_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    
    # Kiểm tra xem có bao nhiêu ảnh đã tồn tại sẵn
    existing = set(os.listdir(out_dir)) if os.path.exists(out_dir) else set()
    needed = [f for f in filenames if f not in existing]
    already_done = len(filenames) - len(needed)
    
    print(f"\n[2/2] Tổng số ảnh: {len(filenames):,} | Đã có sẵn: {already_done:,} | Cần tải: {len(needed):,}")
    print(f"Số luồng tải đồng thời: {concurrency}")

    pbar = tqdm(total=len(filenames), initial=already_done, desc="Tải ảnh COCO", unit="ảnh")

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [download_image(session, semaphore, f, out_dir, pbar) for f in needed]
        await asyncio.gather(*tasks)

    pbar.close()


def run_subset_download(dest_dir, num_images, seed, concurrency):
    t0 = time.time()
    cap_file = download_annotations(dest_dir)
    filenames = get_target_filenames(cap_file, max_pairs=num_images, seed=seed)
    img_dir = os.path.join(dest_dir, "train2017")
    asyncio.run(download_images_async(filenames, img_dir, concurrency=concurrency))
    
    valid_count = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    elapsed = time.time() - t0
    print(f"\n Hoàn tất tải subset COCO 2017:")
    print(f" - Thư mục: {img_dir}")
    print(f" - Số ảnh hợp lệ: {valid_count:,}")
    print(f" - Tổng thời gian: {elapsed:.1f} giây ({elapsed/60:.2f} phút)")


def run_full_download(dest_dir):
    """Tải toàn bộ 18GB bằng aria2c nếu có hoặc Kaggle API"""
    os.makedirs(dest_dir, exist_ok=True)
    print(">>> Đang kiểm tra công cụ tải tối ưu cho file 18GB...")
    
    # Kiểm tra Kaggle
    has_kaggle = shutil.which("kaggle") is not None
    has_aria2c = shutil.which("aria2c") is not None
    
    if has_kaggle:
        print(" Phát hiện Kaggle CLI! Tải qua CDN Google/Kaggle tốc độ cao (50-100 MB/s)...")
        cmd = f"kaggle datasets download -d awsaf49/coco-2017-dataset -p '{dest_dir}' --unzip"
        print(f"Lệnh: {cmd}")
        ret = os.system(cmd)
        if ret == 0:
            print(" Tải qua Kaggle thành công!")
            return

    if has_aria2c:
        print(" Phát hiện aria2c! Tải đa luồng 16 kết nối...")
        cmd_ann = f"aria2c -x 16 -s 16 -k 1M -c -d '{dest_dir}' '{ANNOTATIONS_URL}'"
        cmd_train = f"aria2c -x 16 -s 16 -k 1M -c -d '{dest_dir}' 'http://images.cocodataset.org/zips/train2017.zip'"
        os.system(cmd_ann)
        os.system(cmd_train)
        print("Đang giải nén...")
        os.system(f"cd '{dest_dir}' && unzip -q -n annotations_trainval2017.zip && unzip -q -n train2017.zip")
        return

    print(" Không có Kaggle CLI hoặc aria2c. Tải bằng script shell cải tiến...")
    sh_script = os.path.join(os.path.dirname(__file__), "download_coco.sh")
    os.system(f"bash '{sh_script}' '{dest_dir}'")


def main():
    parser = argparse.ArgumentParser(description="Tải COCO 2017 siêu tốc cho SLQ")
    parser.add_argument("--dest", default="./data/coco2017", help="Thư mục lưu trữ dataset (mặc định: ./data/coco2017)")
    parser.add_argument("--mode", choices=["subset", "full"], default="subset",
                        help="subset: chỉ tải 20k ảnh cần thiết (~2.8GB, 2-3 phút, khuyên dùng). full: tải 18GB.")
    parser.add_argument("--num_images", type=int, default=20000, help="Số ảnh cần tải khi dùng mode=subset (mặc định: 20000)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (mặc định: 42 khớp với SLQ)")
    parser.add_argument("--concurrency", type=int, default=40, help="Số luồng tải đồng thời khi dùng subset (mặc định: 40)")
    args = parser.parse_args()

    if args.mode == "subset":
        run_subset_download(args.dest, args.num_images, args.seed, args.concurrency)
    else:
        run_full_download(args.dest)


if __name__ == "__main__":
    main()
