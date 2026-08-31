"""
Profiling 深度分析脚本：算子耗时分解 + MatMul 细分 + 数据搬运分析

用法:
    python benchmarks/parse_profiling.py --prof_dir <PROF_*目录> [--label v4]

输出:
    1. 算子类型耗时排名
    2. AI Core vs Vector Core 占比
    3. BatchMatMulV2 细分: q/k/v/o_proj, gate/up/down_proj, attn_score
    4. 数据搬运分析: Transpose, GatherV2, ConcatD (含 shape 信息)
    5. AI Core 利用率 + FP16 MAC ratio
    6. 每 token 耗时分解汇总
"""
import csv
import os
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_csv(prof_dir, pattern):
    for root, dirs, files in os.walk(prof_dir):
        for f in sorted(files):
            if pattern in f and f.endswith(".csv"):
                return os.path.join(root, f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Profiling 深度分析")
    parser.add_argument("--prof_dir", type=str, required=True, help="PROF_* 目录路径")
    parser.add_argument("--label", type=str, default="unknown", help="版本标签")
    parser.add_argument("--iterations", type=int, default=0, help="总迭代次数 (0=自动)")
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

    total_time_us = sum(float(r.get("Task Duration(us)", 0)) for r in rows)

    if args.iterations > 0:
        N = args.iterations
    else:
        N = max(sum(1 for r in rows if "embed_tokens" in r.get("Op Name", "")), 1)

    print("=" * 80)
    print(f" Profiling 深度分析: {args.label}")
    print(f" 数据: {summary_csv}")
    print(f" 总 kernel 时间: {total_time_us/1e6:.3f}s ({len(rows)} kernels, {N} iterations)")
    print(f" 每 iteration: {total_time_us/N/1000:.2f}ms")
    print("=" * 80)

    # 1. 算子类型耗时排名
    by_type = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0, "core": ""})
    for r in rows:
        op = r.get("OP Type", "unknown")
        dur = float(r.get("Task Duration(us)", 0))
        core = r.get("Task Type", "")
        by_type[op]["count"] += 1
        by_type[op]["total_us"] += dur
        by_type[op]["max_us"] = max(by_type[op]["max_us"], dur)
        if not by_type[op]["core"]:
            by_type[op]["core"] = core

    print()
    print(">>> 1. 算子类型耗时排名")
    print(f"  {'算子':<28} {'Core':<18} {'次数':>6} {'总耗时(ms)':>10} {'占比':>7} {'均值(us)':>9} {'每token(ms)':>11}")
    print("  " + "-" * 100)
    for op, d in sorted(by_type.items(), key=lambda x: -x[1]["total_us"]):
        ratio = d["total_us"] / total_time_us * 100
        avg_us = d["total_us"] / d["count"]
        per_iter = d["total_us"] / N / 1000
        print(f"  {op:<28} {d['core']:<18} {d['count']:>6} {d['total_us']/1000:>10.2f} {ratio:>6.1f}% {avg_us:>9.1f} {per_iter:>11.3f}")

    # 2. AI Core vs Vector Core
    core_time = defaultdict(float)
    for r in rows:
        core = r.get("Task Type", "unknown")
        dur = float(r.get("Task Duration(us)", 0))
        core_time[core] += dur

    print()
    print(">>> 2. AI Core vs Vector Core")
    for core, t in sorted(core_time.items(), key=lambda x: -x[1]):
        print(f"  {core:<25} {t/1000:>10.2f}ms ({t/total_time_us*100:>5.1f}%)")

    # 3. BatchMatMulV2 细分
    bmm_rows = [r for r in rows if r.get("OP Type") == "BatchMatMulV2"]
    if bmm_rows:
        bmm_total = sum(float(r.get("Task Duration(us)", 0)) for r in bmm_rows)
        categories = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        cat_stats = defaultdict(lambda: {"count": 0, "total_us": 0})
        for r in bmm_rows:
            name = r.get("Op Name", "")
            dur = float(r.get("Task Duration(us)", 0))
            matched = False
            for cat in categories:
                if cat in name:
                    cat_stats[cat]["count"] += 1
                    cat_stats[cat]["total_us"] += dur
                    matched = True
                    break
            if not matched:
                cat_stats["attn_score"]["count"] += 1
                cat_stats["attn_score"]["total_us"] += dur

        print()
        print(f">>> 3. BatchMatMulV2 细分 ({len(bmm_rows)} kernels, {bmm_total/1000:.2f}ms)")
        print(f"  {'子模块':<18} {'次数':>6} {'总耗时(ms)':>10} {'每token(ms)':>11} {'占BMM':>7} {'占总比':>7}")
        print("  " + "-" * 68)
        mlp_us, attn_us = 0, 0
        for cat in categories + ["attn_score"]:
            d = cat_stats[cat]
            if d["count"] > 0:
                per_iter = d["total_us"] / N / 1000
                print(f"  {cat:<18} {d['count']:>6} {d['total_us']/1000:>10.2f} {per_iter:>11.3f} {d['total_us']/bmm_total*100:>6.1f}% {d['total_us']/total_time_us*100:>6.1f}%")
                if cat in ("gate_proj", "up_proj", "down_proj"):
                    mlp_us += d["total_us"]
                else:
                    attn_us += d["total_us"]
        print("  " + "-" * 68)
        print(f"  {'MLP 合计':<18} {'':>6} {mlp_us/1000:>10.2f} {mlp_us/N/1000:>11.3f} {mlp_us/bmm_total*100:>6.1f}% {mlp_us/total_time_us*100:>6.1f}%")
        print(f"  {'Attention 合计':<18} {'':>6} {attn_us/1000:>10.2f} {attn_us/N/1000:>11.3f} {attn_us/bmm_total*100:>6.1f}% {attn_us/total_time_us*100:>6.1f}%")

    # 4. 数据搬运分析
    print()
    print(">>> 4. 数据搬运分析")
    transport_total_us = 0
    for op_type in ["Transpose", "TransData", "GatherV2", "ConcatD"]:
        op_rows = [r for r in rows if r.get("OP Type") == op_type]
        if not op_rows:
            continue
        op_total = sum(float(r.get("Task Duration(us)", 0)) for r in op_rows)
        transport_total_us += op_total
        per_iter = op_total / N / 1000
        print(f"  {op_type}: {len(op_rows)} 次, 总 {op_total/1000:.2f}ms ({op_total/total_time_us*100:.1f}%), 每token {per_iter:.2f}ms")

        seen_shapes = defaultdict(lambda: {"count": 0, "total_us": 0})
        for r in op_rows:
            shape = r.get("Input Shapes", "")[:80]
            dur = float(r.get("Task Duration(us)", 0))
            seen_shapes[shape]["count"] += 1
            seen_shapes[shape]["total_us"] += dur
        for shape, d in sorted(seen_shapes.items(), key=lambda x: -x[1]["total_us"])[:3]:
            print(f"    shape: {shape:<55} ×{d['count']:<4} {d['total_us']/1000:.2f}ms")
    print(f"  数据搬运合计: {transport_total_us/1000:.2f}ms ({transport_total_us/total_time_us*100:.1f}%), 每token {transport_total_us/N/1000:.2f}ms")

    # 5. AI Core 利用率
    if bmm_rows and "aicore_time(us)" in bmm_rows[0]:
        total_aic = sum(float(r.get("aicore_time(us)", 0)) for r in bmm_rows)
        total_dur = sum(float(r.get("Task Duration(us)", 0)) for r in bmm_rows)
        if total_dur > 0:
            print()
            print(">>> 5. BatchMatMulV2 硬件利用率")
            print(f"  Task Duration:       {total_dur/1000:.2f}ms")
            print(f"  AI Core Time:        {total_aic/1000:.2f}ms")
            print(f"  AI Core 利用率:      {total_aic/total_dur*100:.1f}%")
        mac_ratios = [float(r.get("aic_mac_fp16_ratio", 0)) for r in bmm_rows if float(r.get("aic_mac_fp16_ratio", 0)) > 0]
        if mac_ratios:
            print(f"  平均 FP16 MAC ratio: {sum(mac_ratios)/len(mac_ratios)*100:.1f}%")

    # 6. 每 token 耗时分解
    print()
    print(f">>> 6. 每 token 耗时分解 ({total_time_us/N/1000:.2f}ms)")
    print(f"  {'算子':<25} {'耗时(ms)':>10} {'占比':>7}")
    print("  " + "-" * 45)
    for op, d in sorted(by_type.items(), key=lambda x: -x[1]["total_us"])[:10]:
        per_iter = d["total_us"] / N / 1000
        print(f"  {op:<25} {per_iter:>10.2f} {d['total_us']/total_time_us*100:>6.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
