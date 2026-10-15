#!/usr/bin/env python3
import os
import sys
import json
import argparse

def get_runs(project="slq-benchmark-final", entity=None):
    try:
        import wandb
        api = wandb.Api()
        full_proj = f"{entity}/{project}" if entity else project
        runs = api.runs(full_proj)
        data = {}
        for r in runs:
            # Ưu tiên run finished mới nhất
            if r.state == "finished":
                if r.name not in data or r.createdAt > data[r.name]["created"]:
                    data[r.name] = {
                        "id": r.id,
                        "created": r.createdAt,
                        "summary": dict(r.summary),
                        "config": dict(r.config)
                    }
        return data
    except Exception as e:
        print(f"Lỗi kết nối Wandb: {e}")
        return {}


def format_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    header_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in col_widths) + " |"
    body_lines = ["| " + " | ".join(f"{str(cell):<{w}}" for cell, w in zip(row, col_widths)) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "slq-benchmark-final"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument("--out", default="benchmark_report.md")
    args = parser.parse_args()

    data = get_runs(args.project, args.entity)
    if not data:
        print("Không tìm thấy dữ liệu trên Wandb.")
        return

    def get_row(name, label):
        r = data.get(name, {})
        s = r.get("summary", {})
        c = r.get("config", {})
        time_sec = s.get("efficiency/train_time_sec", 0)
        time_str = f"{time_sec/60:.1f}m" if time_sec else "—"
        vram = s.get("efficiency/peak_vram_gb", 0)
        vram_str = f"{vram:.1f}G" if vram else "—"
        params = c.get("total_param_count", s.get("efficiency/trainable_params", 0))
        param_str = f"{params/1e6:.2f}M" if params >= 1e6 else (f"{params/1e3:.1f}K" if params else "—")
        
        c_i2t_1 = s.get("eval/coco_i2t_r1", 0)
        c_i2t_5 = s.get("eval/coco_i2t_r5", 0)
        c_t2i_1 = s.get("eval/coco_t2i_r1", 0)
        f_i2t_1 = s.get("eval/flickr_i2t_r1", 0)
        f_t2i_1 = s.get("eval/flickr_t2i_r1", 0)

        return [
            label,
            param_str,
            time_str,
            vram_str,
            f"{c_i2t_1:.2f}%" if c_i2t_1 else "—",
            f"{c_i2t_5:.2f}%" if c_i2t_5 else "—",
            f"{c_t2i_1:.2f}%" if c_t2i_1 else "—",
            f"{f_i2t_1:.2f}%" if f_i2t_1 else "—",
            f"{f_t2i_1:.2f}%" if f_t2i_1 else "—",
        ]

    headers = ["Phương pháp / Cấu hình", "Params", "Time", "VRAM", "COCO I2T@1", "COCO I2T@5", "COCO T2I@1", "Flickr I2T@1", "Flickr T2I@1"]

    lines = []
    lines.append("# BÁO CÁO TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐỒNG NHẤT (BF16)")
    lines.append(f"Dự án WandB: `{args.project}` | Phần cứng: 4x NVIDIA A100 SXM4 80GB (bfloat16, Batch 256, 5 epochs)\n")

    # Nhóm 1: Baseline & Đối chứng
    lines.append("## 1. Mốc chuẩn SLQ Gốc & Đối chứng")
    rows1 = [
        ["SLQ Gốc (Bài báo công bố)", "19.7K", "—", "—", "61.20%", "84.90%", "48.50%", "84.20%", "74.80%"],
        get_row("exp1_baseline_l1_mean", "exp1_baseline_l1_mean (Reproduce L1 Mean)"),
        get_row("exp0a_backbone_last_token_zeroshot", "exp0a_backbone_last_token_zeroshot (Mốc sàn Zero-shot)"),
        get_row("exp0b_backbone_last_token_linear", "exp0b_backbone_last_token_linear (Table 5 Linear Head)"),
    ]
    lines.append(format_table(headers, rows1) + "\n")

    # Nhóm 2: Ablation Study 1 - Vị trí tiêm (Injection Depth)
    lines.append("## 2. Ablation Study 1: Vị trí tiêm Query (Injection Depth)")
    lines.append("*Cố định: 20 Queries, Layer 24 Mean Pooling (19.7K params)*\n")
    rows2 = [
        get_row("exp1_baseline_l1_mean", "Tầng 1 (SLQ gốc - 0% depth)"),
        get_row("exp2a_lqi8_l24_mean", "Tầng 8 (33.3% depth)"),
        get_row("exp2b_lqi12_l24_mean", "Tầng 12 (50.0% depth)"),
        get_row("exp2c_lqi16_l24_mean", "Tầng 16 (66.7% depth)"),
        get_row("exp2d_lqi20_l24_mean", "Tầng 20 (83.3% depth)"),
    ]
    lines.append(format_table(headers, rows2) + "\n")

    # Nhóm 3: Ablation Study 2 - Chiến lược trích xuất tầng & Head
    lines.append("## 3. Ablation Study 2: Chiến lược trích xuất tầng & Projector Head")
    lines.append("*Cố định: Tiêm tại Tầng 1 (0% depth)*\n")
    rows3 = [
        get_row("exp3a_last2_mean", "2 tầng cuối [23, 24] + Mean Pooling"),
        get_row("exp3b_last4_mean", "4 tầng cuối [21, 22, 23, 24] + Mean Pooling"),
        get_row("exp3c_last4_residual", "4 tầng cuối [21, 22, 23, 24] + Residual Refinement"),
        get_row("exp3d_spaced4_residual", "4 tầng cách nhau [12, 16, 20, 24] + Residual Refinement"),
    ]
    lines.append(format_table(headers, rows3) + "\n")

    # Nhóm 4: Ablation Study 3 - Tỷ lệ phân bổ Prefix vs Suffix
    lines.append("## 4. Ablation Study 3: Phân bố không gian Query (Prefix - Suffix Ratio)")
    lines.append("*Cố định: Tiêm tại Tầng 1, Layer 24 Mean Pooling*\n")
    rows4 = [
        get_row("exp1_baseline_l1_mean", "0 Prefix + 20 Suffix (SLQ gốc)"),
        get_row("exp4a_split_5pre_15suf_mean", "5 Prefix + 15 Suffix"),
        get_row("exp4b_split_10pre_10suf_mean", "10 Prefix + 10 Suffix"),
        get_row("exp4c_split_15pre_5suf_mean", "15 Prefix + 5 Suffix"),
    ]
    lines.append(format_table(headers, rows4) + "\n")

    # Nhóm 5: Champion Runs
    lines.append("## 5. Các mô hình phối hợp Đỉnh cao (Synergy Champions)")
    rows5 = [
        get_row("exp_grand_champion_lqi12_5pre_15suf_last4_residual", "exp_grand_champion_lqi12_5pre_15suf_last4_residual (Grand Champion)"),
    ]
    lines.append(format_table(headers, rows5) + "\n")

    report_content = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Đã xuất báo cáo thành công ra file: {args.out}")
    print("\n" + report_content)


if __name__ == "__main__":
    main()

