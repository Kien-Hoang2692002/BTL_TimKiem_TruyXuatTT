"""
Tái hiện SLQ (Shared Latent Queries) trên InternVL3-1B, chạy được trên 1 GPU T4/P100 (Kaggle/Colab).

Bám sát code gốc https://github.com/CnFaker/SLQ (train/train_internvl_slq.py, eval/eval_internvl_on_coco.py):
  - Ảnh: pad thành hình vuông (màu xám 127) -> resize 448x448 -> chuẩn hóa ImageNet -> 256 token <IMG_CONTEXT>
  - Prompt: "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
  - N query học được gắn vào CUỐI chuỗi, lấy hidden state lớp cuối của N vị trí đó, mean pooling, chuẩn hóa L2
  - Loss: InfoNCE đối xứng, logits = sim / logit_scale (nhiệt độ học được, khởi tạo 0.07)
  - Backbone đóng băng hoàn toàn, chỉ học meta_queries và logit_scale

Thay đổi so với code gốc (để chạy được trên GPU nhỏ):
  - fp16 + gradient checkpointing thay vì bf16 + flash-attn; batch nhỏ hơn (mặc định 64 thay vì 1024)
  - Gọi thẳng Qwen2Model (bỏ lm_head) để không tốn bộ nhớ tính logits trên 151k từ vựng
  - Sửa lỗi code gốc: dòng tạo meta_queries bị comment, mặc định N=1 -> dùng N=20 như bài báo

Ví dụ:
  python slq.py eval  --data flickr --queries init          # zero-shot (query ngẫu nhiên, chưa train)
  python slq.py train --coco_root /kaggle/input/coco-2017-dataset/coco2017 --max_pairs 20000 --out slq_q20.pt
  python slq.py eval  --data flickr,coco --queries slq_q20.pt
  python slq.py eval  --data custom --custom_csv my.csv --image_root imgs/ --queries slq_q20.pt
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_ID = "OpenGVLab/InternVL3-1B"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TEMPLATE = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
IMAGE_PROMPT = TEMPLATE.format(input="<img>" + "<IMG_CONTEXT>" * 256 + "</img>")
HF_TEST_SETS = {
    "flickr": "hf://datasets/nlphuji/flickr_1k_test_image_text_retrieval@refs%2Fconvert%2Fparquet/TEST/test/*.parquet",
    "coco": "hf://datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval@refs%2Fconvert%2Fparquet/TEST/test/*.parquet",
}


# ----------------------------------------------------------------------------- tiền xử lý
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


# ----------------------------------------------------------------------------- mô hình
class SLQEncoder(nn.Module):
    def __init__(self, num_queries=20, device=None, dtype=None):
        super().__init__()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
        from transformers import AutoModel, AutoTokenizer

        self.device, self.dtype = device, dtype
        self.mllm = AutoModel.from_pretrained(
            MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True,
            use_flash_attn=False, trust_remote_code=True,
        ).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
        self.tok.padding_side = "left"
        self.img_ctx_id = self.tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        for p in self.mllm.parameters():
            p.requires_grad = False

        D = self.mllm.language_model.config.hidden_size
        # query giữ ở fp32 để cập nhật ổn định, ép kiểu khi ghép vào chuỗi
        self.meta_queries = nn.Parameter(torch.randn(num_queries, D, device=device) / math.sqrt(D))
        self.logit_scale = nn.Parameter(torch.tensor(0.07, device=device))
        self.backbone = self.mllm.language_model.model  # Qwen2Model, bỏ lm_head

    def _run(self, embeds, mask):
        B = embeds.size(0)
        q = self.meta_queries.to(self.dtype).unsqueeze(0).expand(B, -1, -1)
        embeds = torch.cat([embeds, q], 1)
        mask = torch.cat([mask, mask.new_ones(B, q.size(1))], 1)
        h = self.backbone(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
        h = h[:, -q.size(1):].float().mean(1)
        return F.normalize(h, dim=-1)

    def encode_text(self, texts):
        enc = self.tok([TEMPLATE.format(input=t) for t in texts], return_tensors="pt",
                       padding=True, truncation=True, max_length=128).to(self.device)
        emb = self.backbone.get_input_embeddings()(enc.input_ids)
        return self._run(emb, enc.attention_mask)

    def encode_image(self, pixels):
        B = pixels.size(0)
        enc = self.tok([IMAGE_PROMPT] * B, return_tensors="pt").to(self.device)
        emb = self.backbone.get_input_embeddings()(enc.input_ids)
        with torch.no_grad():  # ViT + projector đóng băng, không cần đồ thị gradient
            vit = self.mllm.extract_feature(pixels.to(self.device, self.dtype))
        emb = emb.clone()
        emb[enc.input_ids == self.img_ctx_id] = vit.reshape(-1, vit.size(-1)).to(emb.dtype)
        return self._run(emb, enc.attention_mask)

    def save_queries(self, path):
        torch.save({"meta_queries": self.meta_queries.detach().cpu(),
                    "logit_scale": self.logit_scale.detach().cpu()}, path)

    def load_queries(self, path):
        ck = torch.load(path, map_location="cpu")
        self.meta_queries.data.copy_(ck["meta_queries"])
        self.logit_scale.data.copy_(ck["logit_scale"])


# ----------------------------------------------------------------------------- dữ liệu
class PairDataset(Dataset):
    def __init__(self, pairs, image_root=""):
        self.pairs, self.root = pairs, image_root

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, cap = self.pairs[i]
        try:
            return preprocess_image(Image.open(os.path.join(self.root, path))), cap
        except Exception as e:
            print("Bỏ qua ảnh lỗi:", path, e)
            return None


def collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def find_coco(coco_root):
    """Tìm captions_train2017.json và thư mục train2017 (Kaggle mount mỗi dataset một kiểu đường dẫn)."""
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
    raise FileNotFoundError(
        f"Không tìm thấy COCO (annotations={ann_path}, train2017={img_dir}). "
        "Hãy Add Input -> tab Datasets -> 'awsaf49/coco-2017-dataset' (dataset, không phải notebook).")


def load_coco_train_pairs(coco_root, max_pairs, seed):
    ann_path, img_dir = find_coco(coco_root)
    print("COCO annotations:", ann_path, "| ảnh:", img_dir)
    ann = json.load(open(ann_path))
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    random.Random(seed).shuffle(ann["annotations"])
    seen, pairs = set(), []
    # mỗi ảnh lấy 1 caption để trong batch không có 2 mẫu dương trùng ảnh
    for a in ann["annotations"]:
        if a["image_id"] in seen:
            continue
        seen.add(a["image_id"])
        pairs.append((os.path.join(img_dir, id2file[a["image_id"]]), a["caption"].strip()))
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def load_test_set(name, args):
    """Trả về (danh sách ảnh PIL hoặc đường dẫn, danh sách caption, img_of_text)."""
    if name in HF_TEST_SETS:
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=HF_TEST_SETS[name], split="train")
        images, texts, img_of_text = [], [], []
        for i, ex in enumerate(ds):
            images.append(ex["image"])
            for c in ex["caption"][:5]:
                texts.append(c)
                img_of_text.append(i)
        return images, texts, img_of_text
    if name == "custom":
        # CSV/JSON gồm 2 cột: image (đường dẫn tương đối tới --image_root), caption.
        # Một ảnh có thể xuất hiện nhiều dòng (nhiều caption).
        import pandas as pd
        f = args.custom_csv
        df = pd.read_json(f) if f.endswith(".json") else pd.read_csv(f)
        uniq = list(dict.fromkeys(df["image"]))
        idx = {p: i for i, p in enumerate(uniq)}
        images = [os.path.join(args.image_root, p) for p in uniq]
        return images, df["caption"].astype(str).tolist(), [idx[p] for p in df["image"]]
    raise ValueError(name)


# ----------------------------------------------------------------------------- train
def train(args):
    torch.manual_seed(args.seed)
    model = SLQEncoder(args.num_queries)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.backbone.train()  # checkpointing chỉ bật khi training=True (Qwen2 không có dropout)

    pairs = load_coco_train_pairs(args.coco_root, args.max_pairs, args.seed)
    print(f"Số cặp huấn luyện: {len(pairs)}")
    dl = DataLoader(PairDataset(pairs), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True, pin_memory=torch.cuda.is_available())

    params = [model.meta_queries, model.logit_scale]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(dl)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    cuda = model.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        for pixels, caps in dl:
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                zi = model.encode_image(pixels)
                zt = model.encode_text(caps)
            logits = zi @ zt.t() / model.logit_scale.clamp(min=0.01)
            labels = torch.arange(len(zi), device=zi.device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} "
                      f"tau {model.logit_scale.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"| {el / 60:.1f} phút, còn ~{el / step * (total - step) / 60:.1f} phút", flush=True)
            if args.save_every and step % args.save_every == 0:
                model.save_queries(args.out)
    model.save_queries(args.out)
    n = sum(p.numel() for p in params)
    print(f"Đã lưu {args.out}. Số tham số huấn luyện: {n:,}")


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def embed_all(model, items, kind, bs):
    out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        with torch.autocast("cuda", dtype=torch.float16, enabled=model.device == "cuda"):
            if kind == "text":
                out.append(model.encode_text(chunk).cpu())
            else:
                imgs = [Image.open(x) if isinstance(x, str) else x for x in chunk]
                out.append(model.encode_image(torch.stack([preprocess_image(im) for im in imgs])).cpu())
        print(f"\r  {kind}: {min(i + bs, len(items))}/{len(items)}", end="", flush=True)
    print()
    return torch.cat(out)


def recall(scores, positive, ks=(1, 5, 10)):
    """scores[q, c]; positive[q, c] = True nếu c là đáp án đúng của q."""
    res = {}
    for k in ks:
        top = scores.topk(min(k, scores.size(1)), dim=1).indices
        res[k] = positive.gather(1, top).any(1).float().mean().item() * 100
    return res


def evaluate(args):
    model = SLQEncoder(args.num_queries)
    if args.queries != "init":
        model.load_queries(args.queries)
    results = {}
    for name in args.data.split(","):
        print(f"== {name}")
        images, texts, img_of_text = load_test_set(name, args)
        print(f"  {len(images)} ảnh, {len(texts)} caption")
        zt = embed_all(model, texts, "text", args.batch_size * 3)
        zi = embed_all(model, images, "image", args.batch_size)
        s = zt @ zi.t()  # [T, I]
        pos = torch.zeros_like(s, dtype=torch.bool)
        pos[torch.arange(len(texts)), torch.tensor(img_of_text)] = True
        t2i, i2t = recall(s, pos), recall(s.t(), pos.t())
        results[name] = {"T2I": t2i, "I2T": i2t}
        print(f"  Image->Text R@1/5/10: {i2t[1]:.1f} / {i2t[5]:.1f} / {i2t[10]:.1f}")
        print(f"  Text->Image R@1/5/10: {t2i[1]:.1f} / {t2i[5]:.1f} / {t2i[10]:.1f}")
    tag = os.path.splitext(os.path.basename(args.queries))[0]
    with open(f"results_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Đã ghi results_{tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--num_queries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    # train
    ap.add_argument("--coco_root", default="/kaggle/input/coco-2017-dataset/coco2017")
    ap.add_argument("--max_pairs", type=int, default=20000, help="0 = toàn bộ ~118k ảnh COCO")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--out", default="slq_q20.pt")
    # eval
    ap.add_argument("--data", default="flickr")
    ap.add_argument("--queries", default="init", help="'init' = query ngẫu nhiên chưa train, hoặc đường dẫn .pt")
    ap.add_argument("--custom_csv")
    ap.add_argument("--image_root", default="")
    args = ap.parse_args()
    train(args) if args.mode == "train" else evaluate(args)
