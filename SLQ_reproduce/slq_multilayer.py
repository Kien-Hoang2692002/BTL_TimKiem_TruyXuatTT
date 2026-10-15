"""
Tái hiện & Cải tiến SLQ: Multi-Layer Aggregation + Projection Head (MLP / Linear / Weighted)
Chạy trên InternVL3-1B, tương thích 1 GPU T4/P100 (Kaggle/Colab) hoặc Ampere+ (L4/A100).

Ý tưởng cải tiến:
  - Thay vì chỉ lấy mean pooling ở layer cuối cùng (Last Layer), mô hình trích xuất hidden states
    của N query tokens từ K layers khác nhau (mặc định 4 layers cuối: -4, -3, -2, -1).
  - Tầng sâu/tầng giữa giữ lại các biểu diễn ngữ nghĩa đa dạng (semantic & syntactic), tránh việc
    bị thiên lệch bởi riêng layer cuối (vốn tối ưu cho next-token prediction).
  - Các query representations từ K layers được tổng hợp qua một Projection Head:
      + 'mlp' (mặc định): Nối K layers [B, K*D] -> Linear(K*D -> D) -> LayerNorm -> GELU -> Linear(D -> D)
      + 'linear': Nối K layers [B, K*D] -> Linear(K*D -> D)
      + 'weighted': Trọng số học được (Softmax) qua K layers (không tăng số chiều)
  - Vẫn giữ nguyên triết lý SLQ: MLLM backbone ĐÓNG BĂNG 100%, chỉ học N query tokens, logit_scale
    và các tham số của Projection Head. Huấn luyện siêu nhanh, không gây catastrophic forgetting.

Ví dụ sử dụng:
  # 1. Zero-shot test (chưa huấn luyện):
  python slq_multilayer.py eval --data flickr --queries init --layers -4,-3,-2,-1 --head_type mlp

  # 2. Huấn luyện với Multi-layer MLP trên COCO 20k cặp:
  python slq_multilayer.py train --coco_root /kaggle/input/coco-2017-dataset/coco2017 \
         --max_pairs 20000 --layers -4,-3,-2,-1 --head_type mlp --out slq_multilayer_q20.pt

  # 3. Đánh giá sau huấn luyện trên Flickr và COCO:
  python slq_multilayer.py eval --data flickr,coco --queries slq_multilayer_q20.pt

  # 4. Huấn luyện biến thể Softmax Weighted Average (cực nhẹ, chỉ thêm K tham số float):
  python slq_multilayer.py train --coco_root /kaggle/input/coco-2017-dataset/coco2017 \
         --layers -4,-3,-2,-1 --head_type weighted --out slq_weighted_q20.pt
"""
import argparse
import json
import math
import os
import random
import sys
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


def load_env():
    """Tự động nạp biến môi trường từ file .env nếu có."""
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
    ]
    for env_path in candidates:
        if os.path.isfile(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
                break
            except Exception:
                pass


load_env()

# Vá lỗi tương thích cho transformers >= 4.49 / 5.x với InternVL (AttributeError: all_tied_weights_keys)
try:
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
    if not hasattr(nn.Module, "all_tied_weights_keys"):
        nn.Module.all_tied_weights_keys = property(lambda self: {})
except Exception:
    pass

# Vá lỗi tương thích cho datasets và huggingface_hub mismatch (BucketNotFoundError)
try:
    import huggingface_hub.errors
    if not hasattr(huggingface_hub.errors, "BucketNotFoundError"):
        class BucketNotFoundError(Exception):
            pass
        huggingface_hub.errors.BucketNotFoundError = BucketNotFoundError
except Exception:
    pass



# ----------------------------------------------------------------------------- tiền xử lý ảnh
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


# ----------------------------------------------------------------------------- các loại Head tổng hợp
class MultiLayerMLPHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class LinearProjectorHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class LearnableLayerWeighting(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        # Khởi tạo trọng số bằng nhau cho mọi tầng (softmax ban đầu là đều)
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_tensors):
        # layer_tensors: list gồm K tensors, mỗi tensor có shape [B, D]
        stacked = torch.stack(layer_tensors, dim=1)  # [B, K, D]
        probs = F.softmax(self.weights, dim=0).view(1, -1, 1)  # [1, K, 1]
        return (stacked * probs).sum(dim=1)  # [B, D]


class CrossAttentionLayerPooling(nn.Module):
    """Cách 3: Query Cross-Attention pooling qua các tầng (Perceiver-style)."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) / math.sqrt(dim))
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, layer_tensors):
        # layer_tensors: list gồm K tensors, mỗi tensor có shape [B, D]
        kv = torch.stack(layer_tensors, dim=1)  # [B, K, D]
        B = kv.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, D]
        attn_out, _ = self.mha(query=q, key=kv, value=kv)  # [B, 1, D]
        x = self.norm1(attn_out + q)
        out = self.norm2(x + self.mlp(x))
        return out.squeeze(1)  # [B, D]


class MeanLayerPooling(nn.Module):
    """Cách a: Mean pooling qua K tầng (không thêm tham số)."""
    def forward(self, layer_tensors):
        return torch.stack(layer_tensors, dim=1).mean(dim=1)


class ResidualRefinementHead(nn.Module):
    """Cách c: Residual Refinement kết hợp tầng cuối cùng với delta từ (K-1) tầng khác.
    Formula: h_final = h_last + alpha * MLP([h_1, ..., h_{K-1}])
    Khởi tạo alpha = 0.0 (Zero-Init Residual).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, layer_tensors):
        # layer_tensors[:-1]: K-1 tầng trước (vd: 3 tầng trước)
        # layer_tensors[-1]: tầng cuối cùng (vd: layer 24)
        h_last = layer_tensors[-1]
        concat_lower = torch.cat(layer_tensors[:-1], dim=-1)
        delta = self.mlp(concat_lower)
        return h_last + self.alpha * delta


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)


# ----------------------------------------------------------------------------- mô hình SLQ Multi-Layer
class SLQMultiLayerEncoder(nn.Module):
    def __init__(self, num_queries=20, layers=(6, 12, 18, 24), head_type="residual", pooling="mean", device=None, dtype=None, attn_implementation="eager", vit_chunk_size=64, inject_layer=1, query_split="none"):
        super().__init__()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            native_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if native_bf16 else torch.float32
        self.inject_layer = max(1, int(inject_layer))
        self.query_split = (query_split or "none").lower()
        print(f"Device: {device} | Dtype: {dtype} | Queries: {num_queries} | InjectLayer: {self.inject_layer} | Split: {self.query_split} | Pooling: {pooling} | Layers: {layers} | Head: {head_type} | Attn: {attn_implementation}")

        from transformers import AutoModel, AutoTokenizer

        self.device, self.dtype = device, dtype
        self.num_queries = num_queries
        self.pooling = (pooling or "mean").lower()
        self.layers = tuple(layers)
        self.head_type = head_type.lower()
        self.vit_chunk_size = vit_chunk_size

        # Nạp MLLM an toàn với cơ chế fallback tự động cho InternVL
        attn_candidates = [attn_implementation]
        if "eager" not in attn_candidates:
            attn_candidates.append("eager")
        attn_candidates.append(None)

        self.mllm = None
        last_err = None
        for low_cpu in [True, False]:
            for attn in attn_candidates:
                kwargs = {
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": low_cpu,
                    "trust_remote_code": True,
                }
                if attn is not None:
                    kwargs["attn_implementation"] = attn
                try:
                    self.mllm = AutoModel.from_pretrained(MODEL_ID, **kwargs).to(device).eval()
                    break
                except Exception as e:
                    last_err = e
                    continue
            if self.mllm is not None:
                break

        if self.mllm is None:
            raise RuntimeError(f"Không thể nạp mô hình {MODEL_ID}: {last_err}")

        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
        self.tok.padding_side = "left"
        self.img_ctx_id = self.tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        for p in self.mllm.parameters():
            p.requires_grad = False

        D = self.mllm.language_model.config.hidden_size
        self.hidden_dim = D

        self.meta_queries = None
        self.query_norm = None
        self.meta_queries_early = None
        self.meta_queries_late = None
        self.query_norm_late = None
        self.meta_queries_prefix = None
        self.meta_queries_suffix = None

        half_q = max(1, num_queries // 2)
        if num_queries == 0:
            self.num_pool_queries = 0
        elif self.query_split == "layer_split":
            # 10 queries ở Layer 1 + 10 queries ở inject_layer (Layer 18). Chỉ pool 10 queries cuối.
            self.meta_queries_early = nn.Parameter(torch.randn(half_q, D, device=device) / math.sqrt(D))
            self.meta_queries_late = nn.Parameter(torch.randn(half_q, D, device=device) / math.sqrt(D))
            self.query_norm_late = nn.LayerNorm(D, device=device)
            self.num_pool_queries = half_q
        elif self.query_split in ("seq_split", "prefix_suffix"):
            # 10 queries ở đầu chuỗi (prefix) + 10 queries ở cuối chuỗi (suffix). Chỉ pool 10 queries cuối.
            self.meta_queries_prefix = nn.Parameter(torch.randn(half_q, D, device=device) / math.sqrt(D))
            self.meta_queries_suffix = nn.Parameter(torch.randn(half_q, D, device=device) / math.sqrt(D))
            self.num_pool_queries = half_q
        else:
            self.meta_queries = nn.Parameter(torch.randn(num_queries, D, device=device) / math.sqrt(D))
            self.query_norm = nn.LayerNorm(D, device=device)
            self.num_pool_queries = num_queries

        self.logit_scale = nn.Parameter(torch.tensor(0.07, device=device))
        self.backbone = self.mllm.language_model.model  # Qwen2Model

        K = len(self.layers)
        if self.head_type in ("none", "identity"):
            self.head = None
        elif self.num_queries == 0:
            # Đối chứng: trích xuất Last Token từ backbone
            if self.head_type == "linear":
                # Giống hệt Table 5 "Head (Linear)" trong bài báo gốc
                self.head = LinearProjectorHead(in_dim=K * D, out_dim=D).to(device)
            elif self.head_type == "mlp":
                self.head = MultiLayerMLPHead(in_dim=K * D, out_dim=D).to(device)
            else:
                self.head = None
        elif K == 1 and self.layers[0] in (-1, self.mllm.language_model.config.num_hidden_layers, 24) and self.head_type in ("mean", "none", "identity"):
            # Nếu chỉ chọn 1 layer cuối, giữ nguyên dạng pooling thuần như SLQ gốc
            self.head = None
        elif self.head_type == "mean":
            self.head = MeanLayerPooling().to(device)
        elif self.head_type == "mlp":
            self.head = MultiLayerMLPHead(in_dim=K * D, out_dim=D).to(device)
        elif self.head_type == "residual":
            self.head = ResidualRefinementHead(in_dim=(K - 1) * D, out_dim=D).to(device)
        elif self.head_type == "linear":
            self.head = LinearProjectorHead(in_dim=K * D, out_dim=D).to(device)
        elif self.head_type == "weighted":
            self.head = LearnableLayerWeighting(num_layers=K).to(device)
        elif self.head_type in ("cross_attn", "attention"):
            self.head = CrossAttentionLayerPooling(dim=D).to(device)
        else:
            raise ValueError(f"head_type không hợp lệ: {head_type}. Chọn 'mean', 'mlp', 'residual', 'linear', 'weighted', 'cross_attn', hoặc 'none'")

    def _run(self, embeds, mask):
        B = embeds.size(0)
        hook_handle = None

        if self.num_queries == 0 or self.num_pool_queries == 0:
            pass
        elif self.query_split == "layer_split":
            # 10 queries ở đầu mạng (Layer 1) + 10 queries ở cuối mạng (Layer inject_layer)
            half_q = self.num_pool_queries
            q_early = self.meta_queries_early.to(self.dtype).unsqueeze(0).expand(B, -1, -1)
            q_late_dummy = torch.zeros(B, half_q, self.hidden_dim, dtype=self.dtype, device=embeds.device)
            embeds = torch.cat([embeds, q_early, q_late_dummy], dim=1)
            mask = torch.cat([mask, mask.new_ones(B, half_q * 2)], dim=1)

            q_late_fresh = self.query_norm_late(self.meta_queries_late.to(self.dtype)).unsqueeze(0).expand(B, -1, -1)

            def inject_hook(module, args):
                h = args[0]
                h_new = torch.cat([h[:, :-half_q], q_late_fresh], dim=1)
                return (h_new,) + args[1:]

            target_layer_idx = min(len(self.backbone.layers) - 1, max(0, self.inject_layer - 1))
            hook_handle = self.backbone.layers[target_layer_idx].register_forward_pre_hook(inject_hook)

        elif self.query_split in ("seq_split", "prefix_suffix"):
            # 10 queries ở đầu chuỗi (prefix) + 10 queries ở cuối chuỗi (suffix)
            q_prefix = self.meta_queries_prefix.to(self.dtype).unsqueeze(0).expand(B, -1, -1)
            q_suffix = self.meta_queries_suffix.to(self.dtype).unsqueeze(0).expand(B, -1, -1)
            embeds = torch.cat([q_prefix, embeds, q_suffix], dim=1)
            mask = torch.cat([mask.new_ones(B, q_prefix.size(1)), mask, mask.new_ones(B, q_suffix.size(1))], dim=1)

        else:
            # Chế độ thông thường
            if self.inject_layer <= 1:
                q = self.meta_queries.to(self.dtype).unsqueeze(0).expand(B, -1, -1)
                embeds = torch.cat([embeds, q], dim=1)
                mask = torch.cat([mask, mask.new_ones(B, q.size(1))], dim=1)
            else:
                q_dummy = torch.zeros(B, self.num_queries, self.hidden_dim, dtype=self.dtype, device=embeds.device)
                embeds = torch.cat([embeds, q_dummy], dim=1)
                mask = torch.cat([mask, mask.new_ones(B, self.num_queries)], dim=1)

                q_fresh = self.query_norm(self.meta_queries.to(self.dtype)).unsqueeze(0).expand(B, -1, -1)

                def inject_hook(module, args):
                    h = args[0]
                    h_new = torch.cat([h[:, :-self.num_queries], q_fresh], dim=1)
                    return (h_new,) + args[1:]

                target_layer_idx = min(len(self.backbone.layers) - 1, max(0, self.inject_layer - 1))
                hook_handle = self.backbone.layers[target_layer_idx].register_forward_pre_hook(inject_hook)

        # Trích xuất hidden states từ toàn bộ các layer
        try:
            outputs = self.backbone(
                inputs_embeds=embeds,
                attention_mask=mask,
                use_cache=False,
                output_hidden_states=True
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        all_hidden_states = outputs.hidden_states  # Tuple gồm (embed_out, layer_0, ..., layer_last)

        # Lấy hidden states ở các layer được chỉ định
        layer_embeds = []
        for l_idx in self.layers:
            # l_idx có thể âm (vd -1 là last layer, -2 là áp chót...)
            h_layer = all_hidden_states[l_idx]
            if self.num_pool_queries > 0:
                # Cắt đúng num_pool_queries vị trí cuối cùng (luôn là 10 tokens cuối chuỗi / cuối tầng)
                q_tokens = h_layer[:, -self.num_pool_queries:]
                if self.pooling == "last":
                    feat = q_tokens[:, -1, :].float()
                elif self.pooling == "max":
                    feat = q_tokens.float().max(dim=1).values
                else:  # "mean" (mặc định của SLQ gốc)
                    feat = q_tokens.float().mean(dim=1)
            else:
                # Không dùng queries: trích xuất last token của chuỗi input trên backbone
                feat = h_layer[:, -1, :].float()
            layer_embeds.append(feat)

        # Tổng hợp qua Head
        if self.head is None:
            h = layer_embeds[0]
        elif self.head_type in ("mean", "weighted", "cross_attn", "attention", "residual"):
            h = self.head(layer_embeds)
        else:  # 'mlp' hoặc 'linear'
            concat_h = torch.cat(layer_embeds, dim=-1)  # [B, K * D]
            h = self.head(concat_h)  # [B, D]

        return F.normalize(h, dim=-1)

    def encode_text(self, texts):
        enc = self.tok([TEMPLATE.format(input=t) for t in texts], return_tensors="pt",
                       padding=True, truncation=True, max_length=128).to(self.device)
        emb = self.backbone.get_input_embeddings()(enc.input_ids)
        return self._run(emb, enc.attention_mask)

    def encode_image_from_vit(self, vit):
        """Chạy trực tiếp từ visual tokens [B, 256, D] mà không cần chạy qua ViT."""
        B = vit.size(0)
        if not hasattr(self, "_cached_img_prompt") or self._cached_img_prompt[0] != B:
            enc = self.tok([IMAGE_PROMPT] * B, return_tensors="pt").to(self.device)
            with torch.no_grad():
                base_emb = self.backbone.get_input_embeddings()(enc.input_ids)
            ctx_mask = (enc.input_ids == self.img_ctx_id)
            self._cached_img_prompt = (B, base_emb, enc.attention_mask, ctx_mask)

        _, base_emb, mask, ctx_mask = self._cached_img_prompt
        emb = base_emb.clone()
        emb[ctx_mask] = vit.to(self.device, self.dtype).reshape(-1, vit.size(-1))
        return self._run(emb, mask)

    def encode_image(self, pixels):
        B = pixels.size(0)
        with torch.no_grad():
            chunk_size = getattr(self, "vit_chunk_size", 64 if self.device == "cuda" else 16)
            vit_list = []
            for i in range(0, B, chunk_size):
                chunk_pixels = pixels[i:i + chunk_size].to(self.device, self.dtype)
                vit_list.append(self.mllm.extract_feature(chunk_pixels))
            vit = torch.cat(vit_list, dim=0)
        return self.encode_image_from_vit(vit)

    def get_trainable_params(self):
        params = []
        for p in [self.meta_queries, self.meta_queries_early, self.meta_queries_late,
                  self.meta_queries_prefix, self.meta_queries_suffix]:
            if p is not None:
                params.append(p)
        for norm in [self.query_norm, self.query_norm_late]:
            if norm is not None:
                params.extend(list(norm.parameters()))
        params.append(self.logit_scale)
        if self.head is not None:
            params.extend(list(self.head.parameters()))
        return params

    def save_queries(self, path):
        ck = {
            "logit_scale": self.logit_scale.detach().cpu(),
            "layers": self.layers,
            "head_type": self.head_type,
            "num_queries": self.num_queries,
            "pooling": self.pooling,
            "inject_layer": getattr(self, "inject_layer", 1),
            "query_split": getattr(self, "query_split", "none"),
        }
        for name in ["meta_queries", "meta_queries_early", "meta_queries_late", "meta_queries_prefix", "meta_queries_suffix"]:
            param = getattr(self, name, None)
            if param is not None:
                ck[name] = param.detach().cpu()
        for name in ["query_norm", "query_norm_late"]:
            norm = getattr(self, name, None)
            if norm is not None:
                ck[f"{name}_state_dict"] = norm.state_dict()
        if self.head is not None:
            ck["head_state_dict"] = self.head.state_dict()
        torch.save(ck, path)

    def load_queries(self, path):
        ck = torch.load(path, map_location="cpu")
        for name in ["meta_queries", "meta_queries_early", "meta_queries_late", "meta_queries_prefix", "meta_queries_suffix"]:
            param = getattr(self, name, None)
            if param is not None and name in ck and ck[name] is not None:
                param.data.copy_(ck[name])
        for name in ["query_norm", "query_norm_late"]:
            norm = getattr(self, name, None)
            if norm is not None and f"{name}_state_dict" in ck:
                norm.load_state_dict(ck[f"{name}_state_dict"])
        if "logit_scale" in ck:
            self.logit_scale.data.copy_(ck["logit_scale"])
        if "head_state_dict" in ck and self.head is not None:
            self.head.load_state_dict(ck["head_state_dict"])
            print("Đã nạp trọng số Head projector thành công.")


# ----------------------------------------------------------------------------- tập dữ liệu & offline caching
class PairDataset(Dataset):
    def __init__(self, pairs, image_root="", cache_dict=None):
        self.pairs = pairs
        self.root = image_root
        self.cache_dict = cache_dict

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, cap = self.pairs[i]
        if self.cache_dict is not None and path in self.cache_dict:
            # ViT token features đã cache sẵn: [256, D]
            return self.cache_dict[path], cap, True
        try:
            full_path = os.path.join(self.root, path) if self.root else path
            img_tensor = preprocess_image(Image.open(full_path))
            return img_tensor, cap, False
        except Exception as e:
            print("Bỏ qua ảnh lỗi:", path, e)
            return None


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    is_cached = batch[0][2]
    items = torch.stack([b[0] for b in batch])
    caps = [b[1] for b in batch]
    return items, caps, is_cached


def precompute_coco_features(model, pairs, cache_path, batch_size=64):
    """
    Trích xuất trước (precompute) ViT features [256, D] cho toàn bộ ảnh huấn luyện và lưu ra file .pt.
    Khi đã có cache, quá trình huấn luyện bỏ qua hoàn toàn Vision Transformer, đạt tốc độ ~10-15x!
    """
    print(f"\n=== Đang trích xuất trước (precompute) ViT features cho {len(pairs)} ảnh ===")
    print(f"File lưu cache: {cache_path}")
    unique_paths = list(dict.fromkeys([p[0] for p in pairs]))
    print(f"Số ảnh duy nhất: {len(unique_paths)}")

    cache_dict = {}
    t0 = time.time()
    model.mllm.eval()

    with torch.no_grad():
        for i in range(0, len(unique_paths), batch_size):
            chunk = unique_paths[i:i + batch_size]
            pixel_list = []
            valid_paths = []
            for p in chunk:
                try:
                    pixel_list.append(preprocess_image(Image.open(p)))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Lỗi đọc ảnh {p}: {e}")
            if not pixel_list:
                continue
            pixels = torch.stack(pixel_list).to(model.device, model.dtype)
            feats = model.mllm.extract_feature(pixels).to(torch.bfloat16).cpu()
            for p, f in zip(valid_paths, feats):
                cache_dict[p] = f
            el = time.time() - t0
            speed = (i + len(chunk)) / max(1e-5, el)
            print(f"\r  Trích xuất: {min(i + batch_size, len(unique_paths))}/{len(unique_paths)} ảnh "
                  f"({speed:.1f} ảnh/s) | Đã chạy {el:.1f}s", end="", flush=True)

    print(f"\nHoàn tất trích xuất trong {time.time() - t0:.1f}s. Đang lưu vào {cache_path}...")
    torch.save(cache_dict, cache_path)
    file_size_gb = os.path.getsize(cache_path) / 1e9
    print(f"Đã lưu cache thành công ({file_size_gb:.2f} GB)!\n")
    return cache_dict


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
    raise FileNotFoundError(
        f"Không tìm thấy COCO (annotations={ann_path}, train2017={img_dir}). "
        "Hãy kiểm tra đường dẫn --coco_root.")


def load_coco_train_pairs(coco_root, max_pairs, seed):
    ann_path, img_dir = find_coco(coco_root)
    print("COCO annotations:", ann_path, "| ảnh:", img_dir)
    ann = json.load(open(ann_path))
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    random.Random(seed).shuffle(ann["annotations"])
    seen, pairs = set(), []
    for a in ann["annotations"]:
        if a["image_id"] in seen:
            continue
        seen.add(a["image_id"])
        pairs.append((os.path.join(img_dir, id2file[a["image_id"]]), a["caption"].strip()))
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def load_test_set(name, args):
    if name in HF_TEST_SETS:
        try:
            import huggingface_hub.errors
            if not hasattr(huggingface_hub.errors, "BucketNotFoundError"):
                class BucketNotFoundError(Exception):
                    pass
                huggingface_hub.errors.BucketNotFoundError = BucketNotFoundError
        except Exception:
            pass
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
        import pandas as pd
        f = args.custom_csv
        df = pd.read_json(f) if f.endswith(".json") else pd.read_csv(f)
        uniq = list(dict.fromkeys(df["image"]))
        idx = {p: i for i, p in enumerate(uniq)}
        images = [os.path.join(args.image_root, p) for p in uniq]
        return images, df["caption"].astype(str).tolist(), [idx[p] for p in df["image"]]
    raise ValueError(name)


# ----------------------------------------------------------------------------- huấn luyện
def train(args):
    # Phát hiện môi trường Distributed Data Parallel (torchrun)
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        import torch.distributed as dist
        import torch.distributed.nn as dist_nn
        from torch.utils.data.distributed import DistributedSampler

        if not dist.is_initialized():
            dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        target_device = f"cuda:{local_rank}"
        if rank == 0:
            print(f"Khởi tạo DDP thành công: {world_size} GPUs tham gia huấn luyện.")
    else:
        rank, local_rank, world_size = 0, 0, 1
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed + rank)
    layers = [int(x.strip()) for x in args.layers.split(",")]
    model = SLQMultiLayerEncoder(
        num_queries=args.num_queries,
        layers=layers,
        head_type=args.head_type,
        pooling=getattr(args, "pooling", "mean"),
        device=target_device,
        attn_implementation=args.attn_implementation,
        inject_layer=getattr(args, "inject_layer", 1),
        query_split=getattr(args, "query_split", "none"),
    )
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.backbone.train()

    pairs = load_coco_train_pairs(args.coco_root, args.max_pairs, args.seed)
    if rank == 0:
        print(f"Số cặp huấn luyện: {len(pairs)}")

    cache_dict = None
    if args.cache_file:
        if os.path.isfile(args.cache_file):
            if rank == 0:
                print(f"Đang nạp ViT feature cache từ {args.cache_file}...")
            t0 = time.time()
            cache_dict = torch.load(args.cache_file, map_location="cpu", weights_only=True)
            if rank == 0:
                print(f"Đã nạp {len(cache_dict)} cached features trong {time.time() - t0:.2f}s!")
        else:
            if rank == 0:
                print(f"Cache file {args.cache_file} chưa tồn tại. Bắt đầu precompute...")
                cache_dict = precompute_coco_features(
                    model, pairs, args.cache_file, batch_size=args.cache_batch_size
                )
            if is_ddp:
                dist.barrier()
            if rank != 0:
                cache_dict = torch.load(args.cache_file, map_location="cpu", weights_only=True)

    sampler = DistributedSampler(PairDataset(pairs, cache_dict=cache_dict), shuffle=True, seed=args.seed) if is_ddp else None
    dl = DataLoader(
        PairDataset(pairs, cache_dict=cache_dict),
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )

    trainable_params = model.get_trainable_params()
    total_param_count = sum(p.numel() for p in trainable_params)
    if rank == 0:
        print(f"Tổng số tham số cần học: {total_param_count:,}")

    # Khởi tạo Wandb nếu được bật (chỉ rank 0)
    wandb_run = None
    if rank == 0 and args.use_wandb and not args.no_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"slq_{args.head_type}_layers{str(args.layers).replace(',', '_')}_bs{args.batch_size}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={
                    "model_id": MODEL_ID,
                    "num_queries": args.num_queries,
                    "pooling": getattr(args, "pooling", "mean"),
                    "layers": args.layers,
                    "head_type": args.head_type,
                    "batch_size": args.batch_size,
                    "world_size": world_size,
                    "effective_batch_size": args.batch_size * world_size * args.grad_accum_steps,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "max_pairs": args.max_pairs,
                    "seed": args.seed,
                    "cache_file": args.cache_file,
                    "attn_implementation": args.attn_implementation,
                    "inject_layer": getattr(args, "inject_layer", 1),
                    "query_split": getattr(args, "query_split", "none"),
                    "total_param_count": total_param_count,
                }
            )
            print(f"Đã kết nối Wandb: Project='{args.wandb_project}' | Run='{run_name}'")
        except Exception as e:
            print(f"Cảnh báo: Không thể khởi tạo Wandb ({e}). Tiếp tục huấn luyện offline.")
            wandb_run = None

    # Áp dụng weight_decay nhỏ cho Head nếu dùng MLP/Linear/CrossAttn/Residual
    if model.head is not None and model.head_type in ("mlp", "linear", "cross_attn", "attention", "residual"):
        head_param_ids = set(id(p) for p in model.head.parameters())
        base_params = [p for p in trainable_params if id(p) not in head_param_ids]
        param_groups = [
            {"params": base_params, "weight_decay": 0.0},
            {"params": list(model.head.parameters()), "weight_decay": 1e-3},
        ]
    else:
        param_groups = [{"params": trainable_params, "weight_decay": 0.0}]

    opt = torch.optim.AdamW(param_groups, lr=args.lr)
    grad_accum = max(1, args.grad_accum_steps)
    steps_per_epoch = math.ceil(len(dl) / grad_accum)
    total_steps = args.epochs * steps_per_epoch
    warm = max(1, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step, micro_step, t0 = 0, 0, time.time()
    opt.zero_grad(set_to_none=True)
    for ep in range(args.epochs):
        if is_ddp and sampler is not None:
            sampler.set_epoch(ep)
        for items, caps, is_cached in dl:
            if is_cached:
                zi = model.encode_image_from_vit(items)
            else:
                zi = model.encode_image(items)
            zt = model.encode_text(caps)
            if not torch.isfinite(zi).all() or not torch.isfinite(zt).all():
                raise RuntimeError("Embedding bị NaN/Inf - dừng huấn luyện")

            if is_ddp and world_size > 1:
                all_zi = torch.cat(dist_nn.all_gather(zi), dim=0)
                all_zt = torch.cat(dist_nn.all_gather(zt), dim=0)
                logits_i2t = zi @ all_zt.t() / model.logit_scale.clamp(min=0.01)
                logits_t2i = zt @ all_zi.t() / model.logit_scale.clamp(min=0.01)
                labels = rank * len(zi) + torch.arange(len(zi), device=zi.device)
                loss = (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels)) / 2
            else:
                logits = zi @ zt.t() / model.logit_scale.clamp(min=0.01)
                labels = torch.arange(len(zi), device=zi.device)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

            loss_scaled = loss / grad_accum
            loss_scaled.backward()
            micro_step += 1

            if micro_step % grad_accum == 0 or micro_step == len(dl):
                if is_ddp and world_size > 1:
                    for p in trainable_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad.data, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if rank == 0:
                    if step % 10 == 0 or step == 1 or step == total_steps:
                        el = time.time() - t0
                        extra_info = ""
                        if model.head_type == "weighted" and model.head is not None:
                            w_probs = F.softmax(model.head.weights, dim=0).detach().cpu().numpy()
                            extra_info = f" weights: {[round(float(w), 3) for w in w_probs]}"
                        elif model.head_type == "residual" and model.head is not None:
                            extra_info = f" alpha: {model.head.alpha.item():.4f}"
                        step_per_sec = step / max(1e-5, el)
                        rem_mins = (total_steps - step) / max(1e-5, step_per_sec) / 60
                        print(f"ep {ep} step {step}/{total_steps} loss {loss.item():.4f} "
                              f"tau {model.logit_scale.item():.4f} lr {sched.get_last_lr()[0]:.2e}{extra_info} "
                              f"({step_per_sec:.1f} steps/s) | {el / 60:.1f}m, còn ~{rem_mins:.1f}m", flush=True)

                    if wandb_run is not None:
                        log_data = {
                            "train/loss": loss.item(),
                            "train/tau": model.logit_scale.item(),
                            "train/lr": sched.get_last_lr()[0],
                            "train/step": step,
                            "train/epoch": ep + (micro_step % len(dl)) / max(1, len(dl)),
                        }
                        if model.head_type == "residual" and model.head is not None:
                            log_data["train/alpha"] = model.head.alpha.item()
                        elif model.head_type == "weighted" and model.head is not None:
                            w_probs = F.softmax(model.head.weights, dim=0).detach().cpu().numpy()
                            for idx, layer_idx in enumerate(model.layers):
                                log_data[f"weights/layer_{layer_idx}"] = float(w_probs[idx])
                        wandb.log(log_data, step=step)

                    if args.save_every and step % args.save_every == 0:
                        model.save_queries(args.out)

    if is_ddp:
        try:
            dist.barrier(device_ids=[local_rank])
        except TypeError:
            dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        train_time = time.time() - t0
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        print(f"\n=== Thống kê hiệu năng ===")
        print(f"  Thời gian huấn luyện : {train_time:.1f}s ({train_time/60:.2f} phút)")
        print(f"  Peak VRAM            : {peak_vram_gb:.2f} GB")
        print(f"  Số tham số học được  : {total_param_count:,}")

        efficiency_info = {
            "num_params": total_param_count,
            "train_time_sec": round(train_time, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
        }

        model.save_queries(args.out)
        print(f"Đã lưu thành công tại {args.out}. Số tham số: {total_param_count:,}")

        if args.eval_after_train:
            print("\n=== Đánh giá mô hình ngay sau huấn luyện ===")
            run_evaluation(model, args, wandb_run=wandb_run, efficiency_info=efficiency_info)

        if wandb_run is not None:
            wandb.summary["efficiency/train_time_sec"] = train_time
            wandb.summary["efficiency/peak_vram_gb"] = peak_vram_gb
            wandb.summary["efficiency/trainable_params"] = total_param_count
            wandb.summary["train/final_loss"] = loss.item()
            wandb.summary["train/total_steps"] = total_steps
            wandb.finish()


# ----------------------------------------------------------------------------- đánh giá
@torch.no_grad()
def embed_all(model, items, kind, bs):
    out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        if kind == "text":
            out.append(model.encode_text(chunk).cpu())
        else:
            imgs = [Image.open(x) if isinstance(x, str) else x for x in chunk]
            out.append(model.encode_image(torch.stack([preprocess_image(im) for im in imgs])).cpu())
        print(f"\r  {kind}: {min(i + bs, len(items))}/{len(items)}", end="", flush=True)
    print()
    out = torch.cat(out)
    bad = (~torch.isfinite(out)).any(1).sum().item()
    if bad:
        raise RuntimeError(f"{bad}/{len(out)} embedding {kind} bị NaN/Inf - kết quả vô nghĩa")
    return out


def recall(scores, positive, ks=(1, 5, 10)):
    res = {}
    for k in ks:
        top = scores.topk(min(k, scores.size(1)), dim=1).indices
        res[k] = positive.gather(1, top).any(1).float().mean().item() * 100
    return res


def run_evaluation(model, args, wandb_run=None, efficiency_info=None):
    results = {}
    for name in args.data.split(","):
        print(f"== Đánh giá tập: {name}")
        images, texts, img_of_text = load_test_set(name, args)
        print(f"  {len(images)} ảnh, {len(texts)} caption")
        zt = embed_all(model, texts, "text", args.batch_size * 2)
        zi = embed_all(model, images, "image", args.batch_size)
        s = zt @ zi.t()  # [T, I]
        pos = torch.zeros_like(s, dtype=torch.bool)
        pos[torch.arange(len(texts)), torch.tensor(img_of_text)] = True
        t2i, i2t = recall(s, pos), recall(s.t(), pos.t())
        results[name] = {"T2I": t2i, "I2T": i2t}
        print(f"  Image->Text R@1/5/10: {i2t[1]:.2f} / {i2t[5]:.2f} / {i2t[10]:.2f}")
        print(f"  Text->Image R@1/5/10: {t2i[1]:.2f} / {t2i[5]:.2f} / {t2i[10]:.2f}")

    if efficiency_info is not None:
        results["efficiency"] = efficiency_info

    if wandb_run is not None:
        import wandb
        eval_metrics = {}
        for ds_name, m in results.items():
            if ds_name == "efficiency":
                continue
            for k in (1, 5, 10):
                eval_metrics[f"eval/{ds_name}_i2t_r{k}"] = m["I2T"][k]
                eval_metrics[f"eval/{ds_name}_t2i_r{k}"] = m["T2I"][k]
        wandb.log(eval_metrics)
        for k, v in eval_metrics.items():
            wandb.summary[k] = v

    tag = os.path.splitext(os.path.basename(args.queries if args.queries != "init" else args.out))[0]
    out_filename = f"results_multilayer_{tag}.json"
    with open(out_filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Đã ghi kết quả đánh giá vào {out_filename}")
    return results


def evaluate(args):
    layers = [int(x.strip()) for x in args.layers.split(",")]
    model = SLQMultiLayerEncoder(
        num_queries=args.num_queries,
        layers=layers,
        head_type=args.head_type,
        pooling=getattr(args, "pooling", "mean"),
        attn_implementation=args.attn_implementation,
        inject_layer=getattr(args, "inject_layer", 1),
        query_split=getattr(args, "query_split", "none"),
    )
    if args.queries != "init":
        model.load_queries(args.queries)

    wandb_run = None
    if args.use_wandb and not args.no_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"eval_{os.path.splitext(os.path.basename(args.queries))[0]}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                job_type="eval",
                config={"queries": args.queries, "layers": args.layers, "head_type": args.head_type, "pooling": getattr(args, "pooling", "mean"), "inject_layer": getattr(args, "inject_layer", 1), "query_split": getattr(args, "query_split", "none")}
            )
        except Exception:
            wandb_run = None

    run_evaluation(model, args, wandb_run=wandb_run)

    if wandb_run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--num_queries", type=int, default=20, help="Số lượng Latent Queries (mặc định 20; đặt 0 để chạy đối chứng Last Token trên backbone)")
    ap.add_argument("--inject_layer", type=int, default=1, help="Tầng bắt đầu chèn Latent Queries (mặc định 1: chèn tại input; hoặc 18, 20, 23)")
    ap.add_argument("--query_split", default="none", choices=["none", "layer_split", "seq_split", "prefix_suffix"], help="Chế độ chia query: 'none' (chuẩn), 'layer_split' (10 tại L1 + 10 tại inject_layer), 'seq_split' (10 prefix + 10 suffix)")
    ap.add_argument("--pooling", default="mean", choices=["mean", "last", "max"], help="Chiến lược pooling qua N queries: 'mean' (mặc định), 'last', hoặc 'max'")
    ap.add_argument("--layers", default="6,12,18,24", help="Các layers cần trích xuất, vd: '6,12,18,24' hoặc '-4,-3,-2,-1'")
    ap.add_argument("--head_type", default="residual", choices=["residual", "mlp", "mean", "cross_attn", "linear", "weighted", "none", "identity"], help="Kiểu projector: 'residual', 'mlp', 'mean', 'linear', 'none', ...")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size trên GPU (khuyến nghị 128 trên A100)")
    ap.add_argument("--grad_accum_steps", type=int, default=8, help="Số bước tích lũy gradient (128 x 8 = 1024 như bài báo gốc)")
    ap.add_argument("--workers", type=int, default=8, help="Số workers DataLoader")
    ap.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"], help="Attention backend (mặc định eager để tương thích InternVL)")
    # train
    ap.add_argument("--coco_root", default="/kaggle/input/coco-2017-dataset/coco2017")
    ap.add_argument("--max_pairs", type=int, default=20000, help="0 = toàn bộ ~118k ảnh COCO")
    ap.add_argument("--epochs", type=int, default=5, help="Số epochs huấn luyện (mặc định 5 như bài báo gốc)")
    ap.add_argument("--lr", type=float, default=5e-4, help="Learning rate (mặc định 5e-4 như bài báo gốc)")
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--out", default="slq_residual_spaced_q20.pt")
    ap.add_argument("--cache_file", default="", help="Đường dẫn file .pt lưu/nạp precomputed ViT features (tăng tốc 10-15x)")
    ap.add_argument("--cache_batch_size", type=int, default=64, help="Batch size khi precompute ViT features trên A100")
    # eval
    ap.add_argument("--data", default="flickr,coco")
    ap.add_argument("--queries", default="init", help="'init' = chưa train, hoặc đường dẫn file .pt")
    ap.add_argument("--custom_csv")
    # wandb logging
    ap.add_argument("--use_wandb", action="store_true", default=bool(os.environ.get("WANDB_API_KEY")),
                    help="Bật logging lên Weights & Biases (tự động bật nếu có WANDB_API_KEY)")
    ap.add_argument("--no_wandb", action="store_true", help="Tắt logging Wandb ngay cả khi có API key")
    ap.add_argument("--wandb_project", default="slq-retrieval", help="Tên Wandb project")
    ap.add_argument("--wandb_entity", default=None, help="Tên user/entity Wandb (tùy chọn)")
    ap.add_argument("--wandb_run_name", default=None, help="Tên run trên Wandb")
    ap.add_argument("--eval_after_train", action="store_true", default=True, help="Tự động đánh giá ngay sau khi train")
    ap.add_argument("--no_eval_after_train", action="store_false", dest="eval_after_train", help="Không đánh giá sau khi train")

    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == "--layers" and sys.argv[i + 1].startswith("-") and not sys.argv[i + 1].startswith("--"):
            sys.argv[i] = f"--layers={sys.argv[i + 1]}"
            del sys.argv[i + 1]
            break
    args = ap.parse_args()

    train(args) if args.mode == "train" else evaluate(args)



