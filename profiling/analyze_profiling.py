"""
Profiling 数据分析脚本
解读 msprof 导出的 CSV，给出优化建议。

用法: python analyze_profiling.py
"""

import os
import csv
import sys

PROF_BASE_DIR = os.path.join(os.path.dirname(__file__), "data")
PROF_DIR = None


def select_prof_dir():
    """查找 PROF_* 目录，多个时提示用户选择"""
    global PROF_DIR
    prof_dirs = []
    for root, dirs, files in os.walk(PROF_BASE_DIR):
        for d in dirs:
            if d.startswith("PROF_"):
                prof_dirs.append(os.path.join(root, d))
    prof_dirs.sort()

    if not prof_dirs:
        print(f"[ERROR] 在 {PROF_BASE_DIR} 下找不到 PROF_* 目录")
        print("请先运行: python profile_inference.py && bash run_profiling.sh --parse")
        sys.exit(1)

    if len(prof_dirs) == 1:
        PROF_DIR = prof_dirs[0]
    else:
        print("[INFO] 发现多个 PROF_* 目录，请选择要分析的目录:")
        for i, d in enumerate(prof_dirs):
            print(f"  [{i + 1}] {d}")
        print()
        try:
            choice = int(input(f"请输入编号 [1-{len(prof_dirs)}]: "))
        except (ValueError, EOFError):
            print("[ERROR] 无效的输入")
            sys.exit(1)
        if choice < 1 or choice > len(prof_dirs):
            print(f"[ERROR] 无效的选择: {choice}")
            sys.exit(1)
        PROF_DIR = prof_dirs[choice - 1]

    print(f"[INFO] 使用 profiling 数据: {PROF_DIR}")
    print()


def find_csv(pattern):
    """在选定的 PROF 目录中找到匹配的 CSV 文件"""
    for root, dirs, files in os.walk(PROF_DIR):
        for f in files:
            if pattern in f and f.endswith(".csv"):
                return os.path.join(root, f)
    return None


def analyze_op_statistic():
    """分析算子统计数据"""
    csv_path = find_csv("op_statistic")
    if not csv_path:
        print("[ERROR] 找不到 op_statistic CSV")
        return

    print("=" * 80)
    print(" 算子耗时统计 (op_statistic)")
    print("=" * 80)
    print(f" 文件: {csv_path}")
    print()

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_time = sum(float(r["Total Time(us)"]) for r in rows)

    print(f"{'算子类型':<30} {'核心':<16} {'次数':>6} {'总耗时(ms)':>10} {'占比':>8} {'平均(us)':>10}")
    print("-" * 90)

    for row in rows:
        op_type = row["OP Type"]
        core = row["Core Type"]
        count = int(row["Count"])
        total_us = float(row["Total Time(us)"])
        avg_us = float(row["Avg Time(us)"])
        ratio = total_us / total_time * 100

        print(f"{op_type:<30} {core:<16} {count:>6} {total_us/1000:>10.2f} {ratio:>7.1f}% {avg_us:>10.3f}")

    print("-" * 90)
    print(f"{'总计':<30} {'':<16} {'':<6} {total_time/1000:>10.2f} {'100.0%':>8}")
    print()


def analyze_api_statistic():
    """分析 ACL API 调用统计"""
    csv_path = find_csv("api_statistic")
    if not csv_path:
        print("[ERROR] 找不到 api_statistic CSV")
        return

    print("=" * 80)
    print(" ACL API 调用统计 (api_statistic)")
    print("=" * 80)
    print(f" 文件: {csv_path}")
    print()

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"{'API 名称':<20} {'调用次数':>8} {'总耗时(ms)':>12} {'平均(ms)':>10} {'最大(ms)':>10}")
    print("-" * 70)
    for row in rows:
        name = row["API Name"]
        count = int(row["Count"])
        total_us = float(row["Time(us)"])
        avg_us = float(row["Avg(us)"])
        max_us = float(row["Max(us)"])
        print(f"{name:<20} {count:>8} {total_us/1000:>12.2f} {avg_us/1000:>10.3f} {max_us/1000:>10.3f}")
    print()


def main():
    if not os.path.exists(PROF_BASE_DIR):
        print(f"[ERROR] profiling 数据目录不存在: {PROF_BASE_DIR}")
        print("请先运行: python profile_inference.py")
        return

    select_prof_dir()
    analyze_op_statistic()
    analyze_api_statistic()


if __name__ == "__main__":
    main()
