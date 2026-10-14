"""
Script tổng hợp kết quả thí nghiệm từ các file JSON trong thư mục hiện tại.
Tự động sinh ra 2 bảng:
1. Bảng Main Results: So sánh Image->Text và Text->Image Recall trên Flickr30K và COCO-5K
2. Bảng Hiệu quả: Số tham số, Thời gian huấn luyện, Peak VRAM
Xuất cả định dạng Markdown và mã HTML sẵn sàng nhúng vào slide báo cáo.
"""
import glob
import json
import os
import sys

EXP_CONFIGS = [
    # Nhóm Đối chứng Backbone (Control Baselines)
    ("Đối chứng: Last Token (Zero-shot)", "results_multilayer_backbone_last_token_zeroshot.json"),
    ("Đối chứng: Last Token (Linear Head)", "results_multilayer_slq_backbone_last_token_linear.json"),
    # Nhóm SLQ Baseline gốc
    ("Baseline SLQ (Layer 24, Mean)", "results_multilayer_slq_baseline_q20.json"),
    # Nhóm 4 layer cuối
    ("4 Layer cuối + Mean", "results_multilayer_slq_last4_mean_q20.json"),
    ("4 Layer cuối + MLP", "results_multilayer_slq_last4_mlp_q20.json"),
    ("4 Layer cuối + Residual", "results_multilayer_slq_last4_residual_q20.json"),
    # Nhóm 4 layer cách đều
    ("4 Layer cách đều + Mean", "results_multilayer_slq_spaced_mean_q20.json"),
    ("4 Layer cách đều + MLP", "results_multilayer_slq_spaced_mlp_q20.json"),
    ("4 Layer cách đều + Residual", "results_multilayer_slq_spaced_residual_q20.json"),
]


def load_result(filename):
    if not os.path.isfile(filename):
        # Thử tìm trong thư mục hiện tại hoặc các thư mục con
        matches = glob.glob(f"**/{filename}", recursive=True)
        if matches:
            filename = matches[0]
        else:
            return None
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    print("=" * 80)
    print("               BẢNG 1: MAIN RESULTS (KẾT QUẢ RETRIEVAL)                 ")
    print("=" * 80)
    header = (
        f"| {'Phương pháp':<32} | {'Flickr I2T R@1':<14} | {'Flickr I2T R@5':<14} | "
        f"{'Flickr T2I R@1':<14} | {'Flickr T2I R@5':<14} | "
        f"{'COCO I2T R@1':<12} | {'COCO T2I R@1':<12} |"
    )
    sep = "|" + "|".join(["-" * (len(col) + 2) for col in header.split("|")[1:-1]]) + "|"
    print(header)
    print(sep)

    all_data = []
    for label, fname in EXP_CONFIGS:
        res = load_result(fname)
        if res is None:
            print(f"| {label:<32} | {'Chưa chạy':<14} | {'-':<14} | {'-':<14} | {'-':<14} | {'-':<12} | {'-':<12} |")
            continue
        flk = res.get("flickr", {})
        coco = res.get("coco", {})
        eff = res.get("efficiency", {})

        f_i2t_1 = flk.get("I2T", {}).get("1", flk.get("I2T", {}).get(1, 0.0))
        f_i2t_5 = flk.get("I2T", {}).get("5", flk.get("I2T", {}).get(5, 0.0))
        f_t2i_1 = flk.get("T2I", {}).get("1", flk.get("T2I", {}).get(1, 0.0))
        f_t2i_5 = flk.get("T2I", {}).get("5", flk.get("T2I", {}).get(5, 0.0))

        c_i2t_1 = coco.get("I2T", {}).get("1", coco.get("I2T", {}).get(1, 0.0))
        c_t2i_1 = coco.get("T2I", {}).get("1", coco.get("T2I", {}).get(1, 0.0))

        row_str = (
            f"| {label:<32} | {f_i2t_1:>13.2f}% | {f_i2t_5:>13.2f}% | "
            f"{f_t2i_1:>13.2f}% | {f_t2i_5:>13.2f}% | "
            f"{c_i2t_1:>11.2f}% | {c_t2i_1:>11.2f}% |"
        )
        print(row_str)
        all_data.append((label, res, eff))

    print("\n" + "=" * 80)
    print("               BẢNG 2: HIỆU NĂNG & CHI PHÍ HUẤN LUYỆN                   ")
    print("=" * 80)
    header2 = f"| {'Phương pháp':<32} | {'Số tham số':<14} | {'Thời gian train':<16} | {'Peak VRAM':<12} |"
    sep2 = "|" + "|".join(["-" * (len(col) + 2) for col in header2.split("|")[1:-1]]) + "|"
    print(header2)
    print(sep2)

    for label, res, eff in all_data:
        n_params = eff.get("num_params", "N/A")
        param_str = f"{n_params:,}" if isinstance(n_params, (int, float)) else str(n_params)
        t_sec = eff.get("train_time_sec", "N/A")
        time_str = f"{t_sec:.1f}s ({t_sec/60:.2f}m)" if isinstance(t_sec, (int, float)) else str(t_sec)
        vram = eff.get("peak_vram_gb", "N/A")
        vram_str = f"{vram:.2f} GB" if isinstance(vram, (int, float)) else str(vram)

        print(f"| {label:<32} | {param_str:>14} | {time_str:>16} | {vram_str:>12} |")

    # Xuất ra file markdown bảng tổng kết
    with open("SUMMARY_TABLES.md", "w", encoding="utf-8") as f:
        f.write("# Bảng Tổng Kết Kết Quả Thí Nghiệm SLQ Multi-Layer\n\n")
        f.write("## 1. Main Results (Hiệu năng truy xuất)\n\n")
        f.write("| Phương pháp | Flickr I2T R@1 | Flickr I2T R@5 | Flickr T2I R@1 | Flickr T2I R@5 | COCO I2T R@1 | COCO T2I R@1 |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for label, fname in EXP_CONFIGS:
            res = load_result(fname)
            if not res:
                continue
            flk, coco = res.get("flickr", {}), res.get("coco", {})
            f_i2t_1 = flk.get("I2T", {}).get("1", flk.get("I2T", {}).get(1, 0.0))
            f_i2t_5 = flk.get("I2T", {}).get("5", flk.get("I2T", {}).get(5, 0.0))
            f_t2i_1 = flk.get("T2I", {}).get("1", flk.get("T2I", {}).get(1, 0.0))
            f_t2i_5 = flk.get("T2I", {}).get("5", flk.get("T2I", {}).get(5, 0.0))
            c_i2t_1 = coco.get("I2T", {}).get("1", coco.get("I2T", {}).get(1, 0.0))
            c_t2i_1 = coco.get("T2I", {}).get("1", coco.get("T2I", {}).get(1, 0.0))
            f.write(f"| **{label}** | {f_i2t_1:.2f}% | {f_i2t_5:.2f}% | {f_t2i_1:.2f}% | {f_t2i_5:.2f}% | {c_i2t_1:.2f}% | {c_t2i_1:.2f}% |\n")

        f.write("\n## 2. Efficiency Comparison (Số tham số, Thời gian, Peak VRAM)\n\n")
        f.write("| Phương pháp | Số tham số học được | Thời gian huấn luyện (1 epoch) | Peak VRAM |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        for label, res, eff in all_data:
            n_params = eff.get("num_params", "N/A")
            param_str = f"{n_params:,}" if isinstance(n_params, (int, float)) else str(n_params)
            t_sec = eff.get("train_time_sec", "N/A")
            time_str = f"{t_sec:.1f}s ({t_sec/60:.2f} phút)" if isinstance(t_sec, (int, float)) else str(t_sec)
            vram = eff.get("peak_vram_gb", "N/A")
            vram_str = f"{vram:.2f} GB" if isinstance(vram, (int, float)) else str(vram)
            f.write(f"| **{label}** | {param_str} | {time_str} | {vram_str} |\n")

    print("\nĐã lưu báo cáo dạng Markdown tại 'SUMMARY_TABLES.md'!")


if __name__ == "__main__":
    main()

