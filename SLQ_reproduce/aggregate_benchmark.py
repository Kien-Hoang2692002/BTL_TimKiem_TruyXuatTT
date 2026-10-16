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
    lines.append("# BÁO CÁO KHOA HỌC: CẢI TIẾN TRUY VẤN TIỀM ẨN THÍCH ỨNG ĐA TẦNG CHO MLLM RETRIEVAL")
    lines.append(f"Dự án WandB: `{args.project}` | Phần cứng: 4x NVIDIA A100 SXM4 80GB (bfloat16, Batch 256, 5 epochs)\n")
    lines.append("> [!NOTE]\n> **Lưu ý về Tốc độ (Wall-clock Time):** Các thực nghiệm ở Mục 1 đến 4 chạy trên GPU sạch (100% idle). Các thực nghiệm Adaptive và Ultimate ở Mục 5, 6 chạy trong phiên sau bị tranh chấp tài nguyên với tiến trình GPU khác trên server, nên thời gian thực thi bị kéo dài hơn. Vì vậy, chỉ nên so sánh thời gian tương đối trong cùng một phiên đo đạc.\n")

    lines.append("## PHẦN I: BÁO CÁO THỰC NGHIỆM CHÍNH THỨC (OFFICIAL BENCHMARK REPORT)\n")

    # 1. Tái lập methods gốc & đối chứng
    lines.append("### 1. Tái lập Phương pháp Gốc & Các Mốc Đối chứng Cơ sở")
    lines.append("*Mục tiêu: Thiết lập mốc sàn tự nhiên của LLM tiền huấn luyện và mốc đối chứng với phương pháp học Linear Head truyền thống.*\n")
    rows1 = [
        get_row("exp0a_backbone_last_token_zeroshot", "Backbone Zero-shot (Last Token)"),
        get_row("exp0b_backbone_last_token_linear", "Backbone + Linear Head (Table 5 Paper)"),
        ["SLQ Gốc (Bài báo công bố)", "19.7K", "—", "—", "61.20%", "84.90%", "48.50%", "84.20%", "74.80%"],
        get_row("exp1_baseline_l1_mean", "SLQ Tái lập (Baseline: 20 Queries L1, L24 Mean)"),
    ]
    lines.append(format_table(headers, rows1) + "\n")

    # 2. Study về prefix prompt
    lines.append("### 2. Nghiên cứu Phân bố Không gian Query (Prefix-Steering Queries)")
    lines.append("*Cố định: Tiêm tại Tầng 1, Layer 24 Mean Pooling. Khảo sát việc phân tách query thành Prefix (đầu chuỗi) và Suffix (đuôi chuỗi).*\n")
    rows2 = [
        get_row("exp1_baseline_l1_mean", "0 Prefix + 20 Suffix (SLQ gốc)"),
        get_row("exp4a_split_5pre_15suf_mean", "5 Prefix + 15 Suffix (Best Prefix Steering) ⭐"),
        get_row("exp4b_split_10pre_10suf_mean", "10 Prefix + 10 Suffix"),
        get_row("exp4c_split_15pre_5suf_mean", "15 Prefix + 5 Suffix"),
    ]
    lines.append(format_table(headers, rows2) + "\n")

    # 3. Study về vị trí chèn prompt
    lines.append("### 3. Nghiên cứu Vị trí Tiêm Query: Tiêm Trễ Tầng Giữa (Late Query Injection - LQI)")
    lines.append("*Cố định: 20 Queries, Layer 24 Mean Pooling (19.7K params). Khảo sát độ sâu lý tưởng để bắt đầu tương tác cross-attention.*\n")
    rows3 = [
        get_row("exp1_baseline_l1_mean", "Tiêm Tầng 1 (0.0% Depth - SLQ gốc)"),
        get_row("exp2a_lqi8_l24_mean", "Tiêm Tầng 8 (33.3% Depth)"),
        get_row("exp2b_lqi12_l24_mean", "Tiêm Tầng 12 (50.0% Depth - LQI-12 Sweet Spot) ⭐"),
        get_row("exp2c_lqi16_l24_mean", "Tiêm Tầng 16 (66.7% Depth)"),
        get_row("exp2d_lqi20_l24_mean", "Tiêm Tầng 20 (83.3% Depth)"),
    ]
    lines.append(format_table(headers, rows3) + "\n")

    # 4. Study về nên lấy feature từ layer nào và tổng hợp thế nào
    lines.append("### 4. Nghiên cứu Trích xuất Đa tầng & Khối Tinh chỉnh Residual (Multi-Layer Residual Refinement)")
    lines.append("*Cố định: Tiêm tại Tầng 1. Khảo sát các chiến lược trích xuất tầng sâu và kiến trúc Projector Head để lọc nhiễu autoregressive.*\n")
    rows4 = [
        get_row("exp3a_last2_mean", "2 tầng cuối [23, 24] + Mean Pooling"),
        get_row("exp3b_last4_mean", "4 tầng cuối [21, 22, 23, 24] + Mean Pooling"),
        get_row("exp3d_spaced4_residual", "4 tầng cách nhau [12, 16, 20, 24] + Residual Head"),
        get_row("exp3c_last4_residual", "4 tầng cuối [21, 22, 23, 24] + Residual Head (MR-4) ⭐"),
    ]
    lines.append(format_table(headers, rows4) + "\n")

    # 5. Study về adapt query prompt
    lines.append("### 5. Nghiên cứu Thích ứng hóa Query theo Đầu vào (Instance-Adaptive Query Generation)")
    lines.append("*Cố định: Tiêm tại Tầng 12, Layer 24 Mean Pooling. Khảo sát khả năng sinh query động từ Last Token của Tầng 11.*\n")
    rows5 = [
        get_row("exp2b_lqi12_l24_mean", "LQI-12 Tĩnh (Static Baseline)"),
        get_row("exp_adaptive_mlp_lqi12_mean", "LQI-12 Thích ứng (Instance-Adaptive MLP Generator - IAQ-12) ⭐"),
    ]
    lines.append(format_table(headers, rows5) + "\n")

    # 6. Bảng kết quả tổng hợp
    lines.append("### 6. Bảng Tổng Hợp So Sánh Toàn Diện & Mô Hình Đề Xuất (Main Benchmark)")
    lines.append("*Tổng hợp các mốc đối chứng, phiên bản tốt nhất của từng trục cải tiến đơn lẻ, và Mô hình Đề xuất Hoàn thiện: **AdaSLQ**.*\n")
    rows6 = [
        get_row("exp0b_backbone_last_token_linear", "Backbone + Linear Head (Table 5 Paper)"),
        ["SLQ Gốc (Bài báo công bố)", "19.7K", "—", "—", "61.20%", "84.90%", "48.50%", "84.20%", "74.80%"],
        get_row("exp1_baseline_l1_mean", "SLQ Tái lập (Baseline)"),
        get_row("exp4a_split_5pre_15suf_mean", "+ Prefix Steering (5 Pre + 15 Suf)"),
        get_row("exp2b_lqi12_l24_mean", "+ Late Query Injection (LQI-12)"),
        get_row("exp3c_last4_residual", "+ Multi-Layer Residual Refinement (MR-4)"),
        get_row("exp_adaptive_mlp_lqi12_mean", "+ Instance-Adaptive Query Generator (IAQ-12)"),
        get_row("exp_ultimate_champion_lqi12_5pre_adaptive_mlp15suf_last4_residual", "AdaSLQ (Mô hình Đề xuất Hoàn thiện: 5-Pre + Adaptive-Suf + MR-4) 👑"),
    ]
    lines.append(format_table(headers, rows6) + "\n")

    # PHẦN II: PHỤ LỤC THỰC NGHIỆM MỞ RỘNG
    lines.append("\n---\n## PHẦN II: PHỤ LỤC THỰC NGHIỆM KỸ THUẬT (TECHNICAL APPENDIX)\n")
    lines.append("### A. Khảo sát Mô hình Phối hợp Trung gian & Ngân hàng Query MoE")
    lines.append("*Lưu vết các thí nghiệm thăm dò kiến trúc khác (Grand V1 tĩnh và DeepSeek MoE Query Pool).*\n")
    rows_app = [
        get_row("exp_grand_champion_lqi12_5pre_15suf_last4_residual", "Grand Champion V1 (5 Pre + 15 Suf Tĩnh + Residual Head: 68.30%)"),
        get_row("exp_adaptive_moe30_lqi12_mean", "Thăm dò: DeepSeek MoE Pool 30 (5 Shared + 15 Routed)"),
        get_row("exp_adaptive_moe60_lqi12_mean", "Thăm dò: DeepSeek MoE Pool 60 (5 Shared + 15 Routed)"),
    ]
    lines.append(format_table(headers, rows_app) + "\n")

    report_content = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Đã xuất báo cáo thành công ra file: {args.out}")
    print("\n" + report_content)


if __name__ == "__main__":
    main()

