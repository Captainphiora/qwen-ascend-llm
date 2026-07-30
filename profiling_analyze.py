"""
Profiling 数据分析脚本：读取 msprof 解析后的 CSV，生成分析报告并保存。

用法:
    python profiling_analyze.py \
        --prof_dir ./profiling_output/xxx/raw/PROF_xxx \
        --output_file ./profiling_output/xxx/analysis_xxx.txt \
        --model_name DeepSeek-R1-Distill-Qwen-1.5B_4096_1 \
        --device_id 0
"""

import argparse
import csv
import os
import sys
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Profiling 数据分析")
    parser.add_argument("--prof_dir", type=str, required=True, help="PROF_* 目录路径")
    parser.add_argument("--output_file", type=str, required=True, help="分析报告输出路径")
    parser.add_argument("--model_name", type=str, default="unknown")
    parser.add_argument("--device_id", type=int, default=0)
    return parser.parse_args()


def find_csv(prof_dir, pattern):
    for root, dirs, files in os.walk(prof_dir):
        for f in files:
            if pattern in f and f.endswith(".csv"):
                return os.path.join(root, f)
    return None


def analyze_op_statistic(prof_dir):
    csv_path = find_csv(prof_dir, "op_statistic")
    if not csv_path:
        return ["[WARN] 未找到 op_statistic CSV", ""]

    lines = []
    lines.append("=" * 80)
    lines.append(" 算子耗时统计 (op_statistic)")
    lines.append("=" * 80)
    lines.append(f" 文件: {csv_path}")
    lines.append("")

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        lines.append(" (无数据)")
        lines.append("")
        return lines

    total_time = sum(float(r.get("Total Time(us)", 0)) for r in rows)

    lines.append(
        f"{'算子类型':<30} {'核心':<16} {'次数':>6} {'总耗时(ms)':>10} {'占比':>8} {'平均(us)':>10}"
    )
    lines.append("-" * 90)

    for row in rows:
        op_type = row.get("OP Type", "N/A")
        core = row.get("Core Type", "N/A")
        count = int(row.get("Count", 0))
        total_us = float(row.get("Total Time(us)", 0))
        avg_us = float(row.get("Avg Time(us)", 0))
        ratio = total_us / total_time * 100 if total_time > 0 else 0
        lines.append(
            f"{op_type:<30} {core:<16} {count:>6} {total_us/1000:>10.2f} {ratio:>7.1f}% {avg_us:>10.3f}"
        )

    lines.append("-" * 90)
    lines.append(f"{'总计':<30} {'':<16} {'':>6} {total_time/1000:>10.2f} {'100.0%':>8}")
    lines.append("")
    return lines


def analyze_api_statistic(prof_dir):
    csv_path = find_csv(prof_dir, "api_statistic")
    if not csv_path:
        return ["[WARN] 未找到 api_statistic CSV", ""]

    lines = []
    lines.append("=" * 80)
    lines.append(" ACL API 调用统计 (api_statistic)")
    lines.append("=" * 80)
    lines.append(f" 文件: {csv_path}")
    lines.append("")

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        lines.append(" (无数据)")
        lines.append("")
        return lines

    lines.append(
        f"{'API 名称':<20} {'调用次数':>8} {'总耗时(ms)':>12} {'平均(ms)':>10} {'最大(ms)':>10}"
    )
    lines.append("-" * 70)

    for row in rows:
        name = row.get("API Name", "N/A")
        count = int(row.get("Count", 0))
        total_us = float(row.get("Time(us)", 0))
        avg_us = float(row.get("Avg(us)", 0))
        max_us = float(row.get("Max(us)", 0))
        lines.append(
            f"{name:<20} {count:>8} {total_us/1000:>12.2f} {avg_us/1000:>10.3f} {max_us/1000:>10.3f}"
        )

    lines.append("")
    return lines


def main():
    args = parse_args()

    if not os.path.isdir(args.prof_dir):
        print(f"[ERROR] PROF 目录不存在: {args.prof_dir}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(" Profiling 分析报告")
    report_lines.append(f" 模型: {args.model_name}")
    report_lines.append(f" 时间: {timestamp}")
    report_lines.append(f" 设备: Device {args.device_id}")
    report_lines.append(f" 数据: {args.prof_dir}")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.extend(analyze_op_statistic(args.prof_dir))
    report_lines.extend(analyze_api_statistic(args.prof_dir))

    report = "\n".join(report_lines)
    print(report)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(report)

    print(f"\n[Analyze] 分析报告已保存: {args.output_file}")


if __name__ == "__main__":
    main()
