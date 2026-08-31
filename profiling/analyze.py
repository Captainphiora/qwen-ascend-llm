#!/usr/bin/env python3
"""
通用 Profiling 分析脚本

功能:
  1. 算子耗时统计 (全量, 不限 Top N)
  2. 数据搬运开销分析 (H2D / D2H / Memcpy)
  3. 数据驱动的瓶颈分析 (不写死任何算子类型)
  4. AI_CPU fallback 检测
  5. Iteration 吞吐统计

用法:
  python3 profiling/analyze.py --prof-dir ./profiling_output/PROF_xxx
  python3 profiling/analyze.py --prof-dir ./profiling_output/PROF_xxx --output report.txt
"""

import argparse
import csv
import glob
import os
import sys
from io import StringIO


def find_csv(base_dir, pattern):
    """在 PROF 目录下递归查找匹配的 CSV 文件"""
    results = sorted(glob.glob(os.path.join(base_dir, "**", pattern), recursive=True))
    return results[-1] if results else None


def read_csv_rows(filepath):
    """读取 CSV 文件返回 list of dict"""
    if not filepath or not os.path.isfile(filepath):
        return []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def format_table(headers, rows, col_widths=None):
    """格式化对齐的文本表格"""
    if not rows:
        return "  (无数据)\n"
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                max_w = max(max_w, len(str(row[i])))
            col_widths.append(min(max_w + 2, 40))

    lines = []
    header_line = "  "
    for i, h in enumerate(headers):
        header_line += h.ljust(col_widths[i])
    lines.append(header_line)
    lines.append("  " + "-" * sum(col_widths))
    for row in rows:
        line = "  "
        for i, val in enumerate(row):
            line += str(val).ljust(col_widths[i])
        lines.append(line)
    return "\n".join(lines) + "\n"


def safe_float(val, default=0.0):
    """安全转换为 float"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def analyze_op_statistic(prof_dir, out):
    """分析算子类型统计 (全量输出)"""
    filepath = find_csv(prof_dir, "op_statistic_*.csv")
    rows = read_csv_rows(filepath)
    if not rows:
        out.write("\n【算子类型耗时统计】\n  (未找到 op_statistic 数据)\n")
        return

    out.write("\n" + "=" * 70 + "\n")
    out.write("【算子类型耗时统计 (全量)】\n")
    out.write(f"  数据来源: {filepath}\n\n")

    time_key = None
    for key in ("Total Time(us)", "Total Duration(us)", "TotalTime(us)"):
        if key in rows[0]:
            time_key = key
            break
    if not time_key:
        for key in rows[0]:
            if "time" in key.lower() or "duration" in key.lower():
                time_key = key
                break
    if not time_key:
        out.write("  [WARN] 无法识别耗时字段\n")
        return

    sorted_rows = sorted(rows, key=lambda r: -safe_float(r.get(time_key, 0)))
    total_us = sum(safe_float(r.get(time_key, 0)) for r in sorted_rows)

    if total_us == 0:
        out.write("  总耗时为 0, 无有效数据\n")
        return

    table_rows = []
    for r in sorted_rows:
        t = safe_float(r.get(time_key, 0))
        op_type = r.get("OP Type", r.get("Op Type", "unknown"))
        core_type = r.get("Core Type", "N/A")
        count = r.get("Count", r.get("count", "N/A"))
        pct = t / total_us * 100
        table_rows.append((op_type, core_type, count, f"{t/1000:.3f}", f"{pct:.2f}%"))

    headers = ["算子类型", "核心类型", "调用次数", "总耗时(ms)", "占比"]
    out.write(format_table(headers, table_rows))
    out.write(f"\n  算子类型总数: {len(sorted_rows)}\n")
    out.write(f"  Device 总耗时: {total_us/1000:.3f} ms\n")

    return sorted_rows, time_key, total_us


def analyze_aicpu_fallback(prof_dir, out):
    """检测 AI_CPU fallback"""
    filepath = find_csv(prof_dir, "op_statistic_*.csv")
    rows = read_csv_rows(filepath)
    if not rows:
        return

    out.write("\n" + "=" * 70 + "\n")
    out.write("【AI_CPU Fallback 检测】\n\n")

    aicpu_rows = [r for r in rows if r.get("Core Type", "") == "AI_CPU"]
    if not aicpu_rows:
        out.write("  [OK] 无 AI_CPU fallback, 所有算子均在 NPU 上执行\n")
    else:
        out.write(f"  [!] 发现 {len(aicpu_rows)} 种算子 fallback 到 AI_CPU:\n\n")
        time_key = None
        for key in ("Total Time(us)", "Total Duration(us)", "TotalTime(us)"):
            if key in aicpu_rows[0]:
                time_key = key
                break
        table_rows = []
        for r in aicpu_rows:
            op_type = r.get("OP Type", r.get("Op Type", "unknown"))
            count = r.get("Count", r.get("count", "N/A"))
            t = safe_float(r.get(time_key, 0)) if time_key else 0
            table_rows.append((op_type, count, f"{t/1000:.3f}"))
        headers = ["算子类型", "调用次数", "总耗时(ms)"]
        out.write(format_table(headers, table_rows))


def analyze_data_transfer(prof_dir, out):
    """分析数据搬运开销 (H2D / D2H / Memcpy)"""
    api_file = find_csv(prof_dir, "api_statistic_*.csv")
    rows = read_csv_rows(api_file)

    out.write("\n" + "=" * 70 + "\n")
    out.write("【数据搬运开销分析 (H2D / D2H)】\n")

    if not rows:
        out.write("  (未找到 api_statistic 数据)\n")
        return

    out.write(f"  数据来源: {api_file}\n\n")

    transfer_keywords = ("memcpy", "memcopy", "copy", "h2d", "d2h", "d2d",
                         "inputcopy", "outputcopy", "transfer")
    transfer_rows = []
    for r in rows:
        name = r.get("Name", r.get("API Name", "")).lower()
        if any(kw in name for kw in transfer_keywords):
            transfer_rows.append(r)

    if not transfer_rows:
        out.write("  未检测到数据搬运相关 API 调用\n")
        all_rows_output(rows, out)
        return

    time_key = None
    for key in ("Total Time(us)", "Total Duration(us)", "TotalTime(us)"):
        if key in transfer_rows[0]:
            time_key = key
            break
    if not time_key:
        for key in transfer_rows[0]:
            if "time" in key.lower() or "duration" in key.lower():
                time_key = key
                break

    table_rows = []
    total_transfer_us = 0
    for r in transfer_rows:
        name = r.get("Name", r.get("API Name", "unknown"))
        count = r.get("Count", r.get("count", "N/A"))
        t = safe_float(r.get(time_key, 0)) if time_key else 0
        total_transfer_us += t
        avg_key = None
        for k in ("Avg Time(us)", "Average Time(us)", "AvgTime(us)"):
            if k in r:
                avg_key = k
                break
        avg_t = safe_float(r.get(avg_key, 0)) if avg_key else 0
        table_rows.append((name, count, f"{t/1000:.3f}", f"{avg_t:.3f}"))

    headers = ["API 名称", "调用次数", "总耗时(ms)", "平均耗时(us)"]
    out.write(format_table(headers, table_rows))
    out.write(f"\n  数据搬运总耗时: {total_transfer_us/1000:.3f} ms\n")

    total_api_us = sum(safe_float(r.get(time_key, 0)) for r in rows) if time_key else 0
    if total_api_us > 0:
        pct = total_transfer_us / total_api_us * 100
        out.write(f"  占 Host API 总耗时比例: {pct:.2f}%\n")


def all_rows_output(rows, out):
    """当没有搬运相关数据时, 输出全部 API 统计供参考"""
    if not rows:
        return
    out.write("\n  所有 Host API 调用统计:\n")
    time_key = None
    for key in ("Total Time(us)", "Total Duration(us)", "TotalTime(us)"):
        if key in rows[0]:
            time_key = key
            break
    sorted_rows = sorted(rows, key=lambda r: -safe_float(r.get(time_key, 0))) if time_key else rows
    table_rows = []
    for r in sorted_rows[:30]:
        name = r.get("Name", r.get("API Name", "unknown"))
        count = r.get("Count", r.get("count", "N/A"))
        t = safe_float(r.get(time_key, 0)) if time_key else 0
        table_rows.append((name, count, f"{t/1000:.3f}"))
    headers = ["API 名称", "调用次数", "总耗时(ms)"]
    out.write(format_table(headers, table_rows))


def analyze_bottleneck(prof_dir, out):
    """数据驱动的瓶颈分析 (不写死算子类型)"""
    summary_file = find_csv(prof_dir, "op_summary_*.csv")
    rows = read_csv_rows(summary_file)

    out.write("\n" + "=" * 70 + "\n")
    out.write("【瓶颈分析 (数据驱动, 基于流水线利用率)】\n")

    if not rows:
        out.write("  (未找到 op_summary 数据)\n")
        return

    out.write(f"  数据来源: {summary_file}\n\n")

    ratio_keys = [k for k in rows[0].keys() if "ratio" in k.lower()]
    if not ratio_keys:
        out.write("  [INFO] op_summary 中无 ratio 字段, 无法进行流水线瓶颈分析\n")
        out.write(f"  可用字段: {list(rows[0].keys())}\n")
        return

    out.write(f"  检测到的 ratio 字段: {ratio_keys}\n\n")

    op_type_key = "OP Type" if "OP Type" in rows[0] else "Op Type"
    op_types = set(r.get(op_type_key, "unknown") for r in rows)

    time_key = None
    for key in ("Task Duration(us)", "Duration(us)", "Total Time(us)"):
        if key in rows[0]:
            time_key = key
            break

    type_stats = {}
    for r in rows:
        op = r.get(op_type_key, "unknown")
        if op not in type_stats:
            type_stats[op] = {"count": 0, "total_time": 0.0, "ratios": {k: [] for k in ratio_keys}}
        type_stats[op]["count"] += 1
        if time_key:
            type_stats[op]["total_time"] += safe_float(r.get(time_key, 0))
        for k in ratio_keys:
            val = safe_float(r.get(k, 0))
            if val > 0:
                type_stats[op]["ratios"][k].append(val)

    sorted_types = sorted(type_stats.items(), key=lambda x: -x[1]["total_time"])

    out.write("  各算子类型流水线利用率 (按总耗时排序):\n\n")
    header = ["算子类型", "次数", "总耗时(ms)"]
    short_ratio_names = []
    for k in ratio_keys:
        short_name = k.replace("_ratio", "").replace("ratio", "").strip("_")
        short_ratio_names.append(short_name)
        header.append(f"avg_{short_name}")
    header.append("瓶颈判断")

    table_rows = []
    bottleneck_summary = {"compute_bound": [], "memory_bound": [], "balanced": []}

    for op, stats in sorted_types:
        row = [op, str(stats["count"]), f"{stats['total_time']/1000:.3f}"]
        avg_ratios = {}
        for k in ratio_keys:
            vals = stats["ratios"][k]
            avg = sum(vals) / len(vals) if vals else 0
            avg_ratios[k] = avg
            row.append(f"{avg:.4f}")

        bottleneck = classify_bottleneck(avg_ratios, ratio_keys)
        row.append(bottleneck)
        table_rows.append(tuple(row))

        if stats["total_time"] > 0:
            if "访存" in bottleneck:
                bottleneck_summary["memory_bound"].append((op, stats["total_time"]))
            elif "计算" in bottleneck:
                bottleneck_summary["compute_bound"].append((op, stats["total_time"]))
            else:
                bottleneck_summary["balanced"].append((op, stats["total_time"]))

    out.write(format_table(header, table_rows))

    out.write("\n  瓶颈总结:\n")
    total_time = sum(s["total_time"] for s in type_stats.values())
    if total_time > 0:
        mem_time = sum(t for _, t in bottleneck_summary["memory_bound"])
        comp_time = sum(t for _, t in bottleneck_summary["compute_bound"])
        bal_time = sum(t for _, t in bottleneck_summary["balanced"])
        out.write(f"    访存瓶颈算子总耗时: {mem_time/1000:.3f} ms ({mem_time/total_time*100:.1f}%)\n")
        out.write(f"    计算瓶颈算子总耗时: {comp_time/1000:.3f} ms ({comp_time/total_time*100:.1f}%)\n")
        out.write(f"    均衡/其他算子总耗时: {bal_time/1000:.3f} ms ({bal_time/total_time*100:.1f}%)\n")

        if mem_time > comp_time and mem_time > bal_time:
            out.write("\n    --> 整体判断: 【访存瓶颈 (Memory Bound)】\n")
            out.write("        大部分耗时算子受限于数据搬运, 建议优化数据布局或减少搬运次数\n")
        elif comp_time > mem_time and comp_time > bal_time:
            out.write("\n    --> 整体判断: 【计算瓶颈 (Compute Bound)】\n")
            out.write("        大部分耗时算子受限于计算单元, 建议优化算子融合或降低计算量\n")
        else:
            out.write("\n    --> 整体判断: 【计算与访存接近平衡】\n")
            out.write("        无明显单一瓶颈, 可从耗时最高的算子逐个优化\n")


def classify_bottleneck(avg_ratios, ratio_keys):
    """根据 ratio 数据判断瓶颈类型 (不写死算子名)"""
    compute_keywords = ("mac", "cube", "vec", "scalar", "aic")
    memory_keywords = ("mte", "mem", "ddr", "hbm", "l2", "ub")

    compute_ratios = []
    memory_ratios = []

    for k, v in avg_ratios.items():
        k_lower = k.lower()
        if any(kw in k_lower for kw in compute_keywords):
            compute_ratios.append(v)
        elif any(kw in k_lower for kw in memory_keywords):
            memory_ratios.append(v)

    avg_compute = sum(compute_ratios) / len(compute_ratios) if compute_ratios else 0
    avg_memory = sum(memory_ratios) / len(memory_ratios) if memory_ratios else 0

    if avg_memory == 0 and avg_compute == 0:
        return "-"
    if avg_memory > avg_compute * 2:
        return "访存瓶颈"
    elif avg_compute > avg_memory * 2:
        return "计算瓶颈"
    else:
        return "均衡"


def analyze_iteration(prof_dir, out):
    """Iteration 吞吐统计"""
    step_file = find_csv(prof_dir, "step_trace_*.csv")
    rows = read_csv_rows(step_file)

    if not rows:
        return

    out.write("\n" + "=" * 70 + "\n")
    out.write("【Iteration 吞吐统计】\n")
    out.write(f"  数据来源: {step_file}\n\n")

    time_key = None
    for key in ("Iteration Time(us)", "IterationTime(us)", "Duration(us)"):
        if key in rows[0]:
            time_key = key
            break
    if not time_key:
        out.write("  [WARN] 无法识别 iteration 耗时字段\n")
        return

    iters = [safe_float(r.get(time_key, 0)) for r in rows if safe_float(r.get(time_key, 0)) > 0]
    if not iters:
        out.write("  无有效 iteration 数据\n")
        return

    steady = iters[3:] if len(iters) > 5 else iters
    avg_iter = sum(steady) / len(steady)
    out.write(f"  总 iteration 数: {len(iters)}\n")
    out.write(f"  稳态平均耗时:    {avg_iter/1000:.3f} ms/iter\n")
    if avg_iter > 0:
        out.write(f"  对应吞吐:        {1e6/avg_iter:.2f} tokens/s\n")
    out.write(f"  最小耗时:        {min(iters)/1000:.3f} ms\n")
    out.write(f"  最大耗时:        {max(iters)/1000:.3f} ms\n")


def analyze_host_api(prof_dir, out):
    """Host API 全量统计"""
    api_file = find_csv(prof_dir, "api_statistic_*.csv")
    rows = read_csv_rows(api_file)

    if not rows:
        return

    out.write("\n" + "=" * 70 + "\n")
    out.write("【Host API 耗时统计 (全量)】\n")
    out.write(f"  数据来源: {api_file}\n\n")

    time_key = None
    for key in ("Total Time(us)", "Total Duration(us)", "TotalTime(us)"):
        if key in rows[0]:
            time_key = key
            break
    if not time_key:
        for key in rows[0]:
            if "time" in key.lower() or "duration" in key.lower():
                time_key = key
                break

    sorted_rows = sorted(rows, key=lambda r: -safe_float(r.get(time_key, 0))) if time_key else rows
    total_us = sum(safe_float(r.get(time_key, 0)) for r in sorted_rows) if time_key else 0

    table_rows = []
    for r in sorted_rows:
        name = r.get("Name", r.get("API Name", "unknown"))
        count = r.get("Count", r.get("count", "N/A"))
        t = safe_float(r.get(time_key, 0)) if time_key else 0
        pct = t / total_us * 100 if total_us > 0 else 0
        table_rows.append((name, count, f"{t/1000:.3f}", f"{pct:.2f}%"))

    headers = ["API 名称", "调用次数", "总耗时(ms)", "占比"]
    out.write(format_table(headers, table_rows))
    out.write(f"\n  Host API 总耗时: {total_us/1000:.3f} ms\n")


def main():
    parser = argparse.ArgumentParser(description="通用 Profiling 分析工具")
    parser.add_argument("--prof-dir", required=True, help="PROF_* 目录路径")
    parser.add_argument("--output", default=None, help="输出报告文件路径 (不指定则仅打印到终端)")
    args = parser.parse_args()

    prof_dir = os.path.realpath(args.prof_dir)
    if not os.path.isdir(prof_dir):
        print(f"[ERROR] 目录不存在: {prof_dir}")
        sys.exit(1)

    buf = StringIO()

    buf.write("=" * 70 + "\n")
    buf.write(" 昇腾 NPU Profiling 分析报告\n")
    buf.write("=" * 70 + "\n")
    buf.write(f"  PROF 目录: {prof_dir}\n")

    analyze_op_statistic(prof_dir, buf)
    analyze_aicpu_fallback(prof_dir, buf)
    analyze_host_api(prof_dir, buf)
    analyze_data_transfer(prof_dir, buf)
    analyze_bottleneck(prof_dir, buf)
    analyze_iteration(prof_dir, buf)

    buf.write("\n" + "=" * 70 + "\n")
    buf.write(" 报告结束\n")
    buf.write("=" * 70 + "\n")

    report = buf.getvalue()
    print(report)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[INFO] 报告已保存至: {args.output}")


if __name__ == "__main__":
    main()
