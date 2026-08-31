"""
Profiling 通用深度分析脚本

完全数据驱动，不写死任何算子名称。从 op_summary CSV 自动发现：
  1. 算子类型耗时排名（全量）
  2. 执行核心分布（AI_CORE / AI_VECTOR_CORE / MIX 等）
  3. Top-1 算子按 Op Name 子图聚类（自动提取最深路径段作为分组键）
  4. 非 AI_CORE 算子详情（数据搬运 / 向量计算），按 shape 聚合
  5. 硬件利用率（如 CSV 包含 aicore_time / mac_ratio 列）
  6. 每 token 耗时分解

用法:
    python benchmarks/parse_profiling.py --prof_dir <PROF_*目录> [--label v4] [--top 15]
"""
import csv
import os
import sys
import re
import argparse
from collections import defaultdict


def find_csv(prof_dir, pattern):
    for root, _, files in os.walk(prof_dir):
        for f in sorted(files):
            if pattern in f and f.endswith(".csv"):
                return os.path.join(root, f)
    return None


def extract_subgraph_key(op_name):
    """从 Op Name 中提取子图分组键。

    例:
      /model/layers.0/self_attn/q_proj/MatMul  → self_attn/q_proj/MatMul
      /model/layers.5/mlp/gate_proj/MatMul      → mlp/gate_proj/MatMul
      PartitionedCall_/model/embed_tokens/Gather → embed_tokens/Gather

    策略: 去掉 layers.N 前缀，保留后面的路径作为分组键。
    如果没有 layers，保留最后两段。
    """
    parts = op_name.replace("PartitionedCall_", "").strip("/").split("/")
    layer_idx = -1
    for i, p in enumerate(parts):
        if re.match(r"layers\.\d+", p):
            layer_idx = i
            break
    if layer_idx >= 0 and layer_idx + 1 < len(parts):
        return "/".join(parts[layer_idx + 1:])
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else op_name


def main():
    parser = argparse.ArgumentParser(description="Profiling 通用深度分析")
    parser.add_argument("--prof_dir", type=str, required=True, help="PROF_* 目录路径")
    parser.add_argument("--label", type=str, default="", help="版本标签")
    parser.add_argument("--iterations", type=int, default=0, help="总迭代次数 (0=自动)")
    parser.add_argument("--top", type=int, default=12, help="各表格显示的最大行数")
    args = parser.parse_args()

    summary_csv = find_csv(args.prof_dir, "op_summary")
    if not summary_csv:
        print(f"[ERROR] 未找到 op_summary CSV in {args.prof_dir}")
        sys.exit(1)

    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[ERROR] op_summary CSV 为空")
        sys.exit(1)

    columns = set(rows[0].keys())
    total_us = sum(float(r.get("Task Duration(us)", 0)) for r in rows)

    if args.iterations > 0:
        N = args.iterations
    else:
        first_op_name = rows[0].get("Op Name", "")
        N = max(sum(1 for r in rows if r.get("Op Name", "") == first_op_name), 1)

    label_str = f": {args.label}" if args.label else ""
    print("=" * 90)
    print(f" Profiling 分析{label_str}")
    print(f" 数据: {summary_csv}")
    print(f" kernel 总数: {len(rows)},  总耗时: {total_us/1e6:.3f}s,  iterations: {N}")
    print(f" 每 iteration: {total_us/N/1000:.2f} ms")
    print("=" * 90)

    # ==================================================================
    # 1. 按 OP Type 聚合
    # ==================================================================
    by_type = defaultdict(lambda: {"count": 0, "us": 0, "max_us": 0, "core": set()})
    for r in rows:
        op = r.get("OP Type", "?")
        dur = float(r.get("Task Duration(us)", 0))
        d = by_type[op]
        d["count"] += 1
        d["us"] += dur
        d["max_us"] = max(d["max_us"], dur)
        core = r.get("Task Type", "")
        if core:
            d["core"].add(core)

    sorted_types = sorted(by_type.items(), key=lambda x: -x[1]["us"])

    print()
    print(">>> 1. 算子类型耗时排名")
    hdr = f"  {'OP Type':<30} {'Core':<18} {'Count':>6} {'Total(ms)':>10} {'Ratio':>7} {'Avg(us)':>9} {'Per-Iter(ms)':>12}"
    print(hdr)
    print("  " + "-" * len(hdr))
    for op, d in sorted_types:
        cores = ",".join(sorted(d["core"])) if d["core"] else "-"
        if len(cores) > 16:
            cores = cores[:14] + ".."
        print(f"  {op:<30} {cores:<18} {d['count']:>6} {d['us']/1000:>10.2f}"
              f" {d['us']/total_us*100:>6.1f}% {d['us']/d['count']:>9.1f}"
              f" {d['us']/N/1000:>12.3f}")

    # ==================================================================
    # 2. 按执行核心聚合
    # ==================================================================
    core_us = defaultdict(float)
    core_cnt = defaultdict(int)
    for r in rows:
        c = r.get("Task Type", "unknown")
        dur = float(r.get("Task Duration(us)", 0))
        core_us[c] += dur
        core_cnt[c] += 1

    print()
    print(">>> 2. 执行核心分布")
    for c, t in sorted(core_us.items(), key=lambda x: -x[1]):
        print(f"  {c:<25} {core_cnt[c]:>6} kernels  {t/1000:>10.2f}ms  ({t/total_us*100:>5.1f}%)")

    # ==================================================================
    # 3. 耗时最高的 OP Type — 按子图名称聚类
    # ==================================================================
    top1_op = sorted_types[0][0] if sorted_types else None
    if top1_op:
        top1_rows = [r for r in rows if r.get("OP Type") == top1_op]
        top1_total = sum(float(r.get("Task Duration(us)", 0)) for r in top1_rows)

        subgraph = defaultdict(lambda: {"count": 0, "us": 0})
        for r in top1_rows:
            key = extract_subgraph_key(r.get("Op Name", ""))
            dur = float(r.get("Task Duration(us)", 0))
            subgraph[key]["count"] += 1
            subgraph[key]["us"] += dur

        print()
        print(f">>> 3. {top1_op} 子图聚类 ({len(top1_rows)} kernels, {top1_total/1000:.2f}ms)")
        hdr = f"  {'子图':<45} {'Count':>6} {'Total(ms)':>10} {'Per-Iter(ms)':>12} {'Ratio':>7}"
        print(hdr)
        print("  " + "-" * len(hdr))
        for key, d in sorted(subgraph.items(), key=lambda x: -x[1]["us"])[:args.top]:
            print(f"  {key:<45} {d['count']:>6} {d['us']/1000:>10.2f}"
                  f" {d['us']/N/1000:>12.3f} {d['us']/top1_total*100:>6.1f}%")

    # ==================================================================
    # 4. 非 Top-1 算子详情 — 按 (OP Type, Input Shape) 聚合
    # ==================================================================
    non_top1 = [r for r in rows if r.get("OP Type") != top1_op] if top1_op else rows
    non_top1_total = sum(float(r.get("Task Duration(us)", 0)) for r in non_top1)

    shape_agg = defaultdict(lambda: {"count": 0, "us": 0})
    for r in non_top1:
        op = r.get("OP Type", "?")
        shape = r.get("Input Shapes", "-")[:70]
        dur = float(r.get("Task Duration(us)", 0))
        key = f"{op}  |  {shape}"
        shape_agg[key]["count"] += 1
        shape_agg[key]["us"] += dur

    print()
    print(f">>> 4. 非 {top1_op} 算子详情 — 按 (OP Type, Shape) 聚合")
    print(f"    合计: {non_top1_total/1000:.2f}ms ({non_top1_total/total_us*100:.1f}%)")
    hdr = f"  {'OP Type  |  Input Shape':<75} {'Count':>6} {'Total(ms)':>10} {'Per-Iter(ms)':>12}"
    print(hdr)
    print("  " + "-" * len(hdr))
    for key, d in sorted(shape_agg.items(), key=lambda x: -x[1]["us"])[:args.top]:
        print(f"  {key:<75} {d['count']:>6} {d['us']/1000:>10.2f} {d['us']/N/1000:>12.3f}")

    # ==================================================================
    # 5. 硬件利用率（如果 CSV 包含相关列）
    # ==================================================================
    has_aic_time = "aicore_time(us)" in columns
    has_aiv_time = "aiv_time(us)" in columns
    has_mac = "aic_mac_fp16_ratio" in columns

    if has_aic_time or has_aiv_time or has_mac:
        print()
        print(">>> 5. 硬件利用率")

        if has_aic_time:
            aic_by_type = defaultdict(lambda: {"task_us": 0, "aic_us": 0})
            for r in rows:
                aic = float(r.get("aicore_time(us)", 0))
                if aic > 0:
                    op = r.get("OP Type", "?")
                    dur = float(r.get("Task Duration(us)", 0))
                    aic_by_type[op]["task_us"] += dur
                    aic_by_type[op]["aic_us"] += aic
            if aic_by_type:
                print(f"  {'OP Type':<30} {'Task(ms)':>10} {'AIC(ms)':>10} {'AIC占比':>8}")
                print("  " + "-" * 62)
                for op, d in sorted(aic_by_type.items(), key=lambda x: -x[1]["task_us"]):
                    ratio = d["aic_us"] / d["task_us"] * 100 if d["task_us"] > 0 else 0
                    print(f"  {op:<30} {d['task_us']/1000:>10.2f} {d['aic_us']/1000:>10.2f} {ratio:>7.1f}%")

        if has_mac:
            mac_by_type = defaultdict(list)
            for r in rows:
                mac = float(r.get("aic_mac_fp16_ratio", 0))
                if mac > 0:
                    mac_by_type[r.get("OP Type", "?")].append(mac)
            if mac_by_type:
                print()
                print(f"  {'OP Type':<30} {'Avg MAC%':>10} {'Max MAC%':>10} {'Samples':>8}")
                print("  " + "-" * 62)
                for op, vals in sorted(mac_by_type.items(), key=lambda x: -sum(x[1])/len(x[1])):
                    print(f"  {op:<30} {sum(vals)/len(vals)*100:>9.1f}% {max(vals)*100:>9.1f}% {len(vals):>8}")

    # ==================================================================
    # 6. 每 iteration 耗时汇总
    # ==================================================================
    print()
    per_iter_ms = total_us / N / 1000
    print(f">>> 6. 每 iteration 耗时汇总: {per_iter_ms:.2f} ms")
    print(f"  {'OP Type':<30} {'Per-Iter(ms)':>12} {'Ratio':>7}")
    print("  " + "-" * 52)
    for op, d in sorted_types[:args.top]:
        pi = d["us"] / N / 1000
        print(f"  {op:<30} {pi:>12.3f} {d['us']/total_us*100:>6.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
