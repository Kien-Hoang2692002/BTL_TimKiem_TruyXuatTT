"""
Script trích xuất trước (offline precomputing) ViT features cho tập dữ liệu COCO.
Giúp quá trình huấn luyện SLQ bỏ qua hoàn toàn Vision Transformer, tăng tốc từ 10x đến 15x trên GPU A100.

Sử dụng:
  python precompute_vit.py --coco_root /path/to/coco2017 --max_pairs 20000 --out coco_vit_20k.pt --batch_size 64
"""
import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from transformers import AutoModel

# Vá lỗi tương thích cho transformers >= 4.49 / 5.x với InternVL (all_tied_weights_keys)
try:
    import torch.nn as nn
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
    if not hasattr(nn.Module, "all_tied_weights_keys"):
        nn.Module.all_tied_weights_keys = property(lambda self: {})
except Exception:
    pass

ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL_ID = "OpenGVLab/InternVL3-1B"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def expand2square(img, bg=(127, 127, 127)):
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = Image.new(img.mode, (side, side), bg)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


def preprocess_image(img):
    img = expand2square(img.convert("RGB"))
    x = torch.from_numpy(np.array(img)).float() / 255
    x = (x - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    x = x.permute(2, 0, 1)[None]
    return F.interpolate(x, size=(448, 448), mode="bilinear")[0]


def find_coco(coco_root):
    ann_path = img_dir = None
    for base in [coco_root, "/kaggle/input"]:
        if not base or not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            if ann_path is None and "captions_train2017.json" in files:
                ann_path = os.path.join(root, "captions_train2017.json")
            if img_dir is None and os.path.basename(root) == "train2017" and any(f.endswith(".jpg") for f in files[:50]):
                img_dir = root
            if ann_path and img_dir:
                return ann_path, img_dir
    raise FileNotFoundError(f"Không tìm thấy COCO trong {coco_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", required=True, help="Đường dẫn thư mục coco2017")
    parser.add_argument("--max_pairs", type=int, default=20000, help="Số lượng ảnh cần trích xuất (0 = toàn bộ)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size trích xuất trên A100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="coco_vit_20k.pt", help="File đầu ra lưu features")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    print(f"Khởi tạo InternVL3-1B trên {device} ({dtype})...")
    try:
        mllm = AutoModel.from_pretrained(
            MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(device).eval()
    except Exception as e:
        print(f"Cảnh báo: load với low_cpu_mem_usage thất bại ({e}), chuyển sang fallback low_cpu_mem_usage=False...")
        mllm = AutoModel.from_pretrained(
            MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=False, trust_remote_code=True
        ).to(device).eval()

    ann_path, img_dir = find_coco(args.coco_root)
    ann = json.load(open(ann_path))
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    random.Random(args.seed).shuffle(ann["annotations"])

    seen, paths = set(), []
    for a in ann["annotations"]:
        if a["image_id"] in seen:
            continue
        seen.add(a["image_id"])
        paths.append(os.path.join(img_dir, id2file[a["image_id"]]))
        if args.max_pairs and len(paths) >= args.max_pairs:
            break

    print(f"Tổng số ảnh duy nhất cần trích xuất: {len(paths)}")
    cache_dict = {}
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            chunk = paths[i:i + args.batch_size]
            pixel_list, valid_paths = [], []
            for p in chunk:
                try:
                    pixel_list.append(preprocess_image(Image.open(p)))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Lỗi đọc {p}: {e}")
            if not pixel_list:
                continue

            pixels = torch.stack(pixel_list).to(device, dtype)
            feats = mllm.extract_feature(pixels).to(torch.bfloat16).cpu()
            for p, f in zip(valid_paths, feats):
                cache_dict[p] = f

            el = time.time() - t0
            speed = (i + len(chunk)) / max(1e-5, el)
            print(f"\r  Tiến độ: {min(i + args.batch_size, len(paths))}/{len(paths)} ảnh "
                  f"({speed:.1f} ảnh/s) | Đã chạy {el:.1f}s", end="", flush=True)

    print(f"\nHoàn tất trích xuất trong {time.time() - t0:.1f}s. Đang lưu vào {args.out}...")
    torch.save(cache_dict, args.out)
    size_gb = os.path.getsize(args.out) / 1e9
    print(f"Đã lưu thành công {len(cache_dict)} features ({size_gb:.2f} GB) tại {args.out}!")


if __name__ == "__main__":
    main()
