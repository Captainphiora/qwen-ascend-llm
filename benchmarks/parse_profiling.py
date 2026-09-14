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

    # ==================================================================
    # 7. HBM 搬运量分析 (Gather 修正)
    # ==================================================================
    GATHER_OPS = {'GatherV2', 'Gather', 'GatherElements', 'GatherND',
                  'EmbeddingLookup'}
    has_shapes = "Input Shapes" in columns and "Input Data Types" in columns
    if has_shapes:
        print()
        print(">>> 7. HBM 搬运量分析 (Gather 类算子按 output size 修正)")

        def parse_shape(s):
            s = s.strip().strip('"')
            if not s or s == 'N/A':
                return ()
            try:
                return tuple(int(x) for x in s.split(',') if x.strip())
            except ValueError:
                return ()

        def numel(shape):
            r = 1
            for d in shape:
                r *= d
            return r

        def dtype_bytes(d):
            d = d.strip().upper()
            if 'FLOAT16' in d: return 2
            if 'FLOAT' in d: return 4
            if 'INT8' in d: return 1
            if 'INT32' in d: return 4
            if 'INT64' in d: return 8
            return 2

        op_mem = defaultdict(float)
        shape_mem = defaultdict(lambda: {"mem": 0, "count": 0})
        total_mem = 0
        total_mem_raw = 0

        for r in rows:
            op_type = r.get("OP Type", "?")
            ins = r.get("Input Shapes", "")
            outs = r.get("Output Shapes", "")
            idts = r.get("Input Data Types", "")
            odts = r.get("Output Data Types", "")
            in_shapes = [parse_shape(s) for s in ins.split(';') if s.strip()]
            out_shapes = [parse_shape(s) for s in outs.split(';') if s.strip()]
            in_dtypes = [s.strip() for s in idts.split(';') if s.strip()]
            out_dtypes = [s.strip() for s in odts.split(';') if s.strip()]

            mem_raw = 0
            for sh, dt in zip(in_shapes, in_dtypes):
                if sh:
                    mem_raw += numel(sh) * dtype_bytes(dt)
            for sh, dt in zip(out_shapes, out_dtypes):
                if sh:
                    mem_raw += numel(sh) * dtype_bytes(dt)
            total_mem_raw += mem_raw

            if op_type in GATHER_OPS:
                mem = 0
                for sh, dt in zip(in_shapes[1:], in_dtypes[1:]):
                    if sh:
                        mem += numel(sh) * dtype_bytes(dt)
                for sh, dt in zip(out_shapes, out_dtypes):
                    if sh:
                        mem += numel(sh) * dtype_bytes(dt) * 2
            else:
                mem = mem_raw

            op_mem[op_type] += mem
            total_mem += mem
            key = (op_type, ins.strip()[:60])
            shape_mem[key]["mem"] += mem
            shape_mem[key]["count"] += 1

        correction_gb = (total_mem_raw - total_mem) / 1e9
        print(f"  修正后搬运: {total_mem/1e9:.3f} GB,  每 iteration: {total_mem/N/1e6:.1f} MB")
        if correction_gb > 0.001:
            print(f"  (修正前声明: {total_mem_raw/1e9:.3f} GB, Gather 修正减去 {correction_gb:.2f} GB)")
        print()
        print(f"  按算子类型:")
        print(f"  {'OP Type':<30} {'Total(GB)':>10} {'Per-Iter(MB)':>13} {'占比':>7}")
        print("  " + "-" * 64)
        for op, mem in sorted(op_mem.items(), key=lambda x: -x[1]):
            if mem / total_mem < 0.001:
                continue
            print(f"  {op:<30} {mem/1e9:>10.3f} {mem/N/1e6:>13.1f} {mem/total_mem*100:>6.1f}%")

        print()
        print(f"  按 (算子, Shape) Top-{min(args.top, 10)}:")
        print(f"  {'OP Type':<25} {'Input Shape':<40} {'Cnt':>5} {'Per-Iter(MB)':>13} {'占比':>7}")
        print("  " + "-" * 95)
        for (op, shape), d in sorted(shape_mem.items(), key=lambda x: -x[1]["mem"])[:min(args.top, 10)]:
            print(f"  {op:<25} {shape:<40} {d['count']:>5} {d['mem']/N/1e6:>13.1f} {d['mem']/total_mem*100:>6.1f}%")

    # ==================================================================
    # 8. task-memory 实际内存 (如果有 task_memory 列)
    # ==================================================================
    mem_cols = [c for c in columns if 'memory' in c.lower() or 'workspace' in c.lower()]
    if mem_cols:
        print()
        print(f">>> 8. task-memory 实测内存")
        print(f"  检测到内存列: {', '.join(mem_cols)}")
        for mc in mem_cols:
            op_task_mem = defaultdict(lambda: {"total": 0, "count": 0, "max": 0})
            for r in rows:
                val = r.get(mc, "")
                if val and val != "N/A":
                    try:
                        v = float(val)
                    except ValueError:
                        continue
                    if v > 0:
                        op = r.get("OP Type", "?")
                        op_task_mem[op]["total"] += v
                        op_task_mem[op]["count"] += 1
                        op_task_mem[op]["max"] = max(op_task_mem[op]["max"], v)
            if op_task_mem:
                print()
                print(f"  [{mc}]:")
                print(f"  {'OP Type':<30} {'Total(MB)':>10} {'Per-Iter(MB)':>13} {'Max(MB)':>10} {'Cnt':>6}")
                print("  " + "-" * 74)
                for op, d in sorted(op_task_mem.items(), key=lambda x: -x[1]["total"])[:args.top]:
                    print(f"  {op:<30} {d['total']/1e6:>10.2f} {d['total']/N/1e6:>13.3f}"
                          f" {d['max']/1e6:>10.3f} {d['count']:>6}")

    # ==================================================================
    # 9. AIC/AIV pipe 详细分解 (mte2 读写占比)
    # ==================================================================
    has_mte2 = "aic_mte2_ratio" in columns or "aic_mte2_time(us)" in columns
    has_aiv_mte2 = "aiv_mte2_ratio" in columns
    if has_mte2 or has_aiv_mte2:
        print()
        print(">>> 9. 数据搬运 (MTE2) 详细分解")
        pipe_by_type = defaultdict(lambda: {
            "dur_us": 0, "aic_us": 0, "aiv_us": 0,
            "aic_mac_us": 0, "aic_mte2_us": 0, "aic_mte1_us": 0,
            "aiv_vec_us": 0, "aiv_mte2_us": 0, "aiv_mte3_us": 0,
        })
        for r in rows:
            op = r.get("OP Type", "?")
            p = pipe_by_type[op]
            dur = float(r.get("Task Duration(us)", 0))
            p["dur_us"] += dur
            aic = float(r.get("aicore_time(us)", 0) or 0)
            aiv = float(r.get("aiv_time(us)", 0) or 0)
            p["aic_us"] += aic
            p["aiv_us"] += aiv
            if aic > 0:
                p["aic_mac_us"] += float(r.get("aic_mac_ratio", 0) or 0) * aic
                p["aic_mte2_us"] += float(r.get("aic_mte2_ratio", 0) or 0) * aic
                p["aic_mte1_us"] += float(r.get("aic_mte1_ratio", 0) or 0) * aic
            if aiv > 0:
                p["aiv_vec_us"] += float(r.get("aiv_vec_ratio", 0) or 0) * aiv
                p["aiv_mte2_us"] += float(r.get("aiv_mte2_ratio", 0) or 0) * aiv
                p["aiv_mte3_us"] += float(r.get("aiv_mte3_ratio", 0) or 0) * aiv

        print(f"  {'OP Type':<25} {'Task(ms)':>8} {'AIC(ms)':>8} {'MAC(ms)':>8} {'MTE2(ms)':>9}"
              f" {'AIV(ms)':>8} {'VEC(ms)':>8} {'MTE2v(ms)':>9}")
        print("  " + "-" * 100)
        for op, p in sorted(pipe_by_type.items(), key=lambda x: -x[1]["dur_us"])[:args.top]:
            print(f"  {op:<25} {p['dur_us']/1000:>8.2f} {p['aic_us']/1000:>8.2f}"
                  f" {p['aic_mac_us']/1000:>8.2f} {p['aic_mte2_us']/1000:>9.2f}"
                  f" {p['aiv_us']/1000:>8.2f} {p['aiv_vec_us']/1000:>8.2f}"
                  f" {p['aiv_mte2_us']/1000:>9.2f}")

    # ==================================================================
    # 10. Host 侧 API 统计 (aclrtMemcpy / ModelExecute)
    # ==================================================================
    api_csv = find_csv(args.prof_dir, "api_statistic")
    if api_csv:
        with open(api_csv) as f:
            api_rows = list(csv.DictReader(f))
        if api_rows:
            print()
            print(">>> 10. Host 侧 API 统计 (aclrtMemcpy / ModelExecute)")
            key_apis = {}
            for r in api_rows:
                name = r.get("API Name", "")
                if not name:
                    continue
                api_total_us = float(r.get("Time(us)", 0))
                count = int(r.get("Count", 0))
                avg_us = float(r.get("Avg(us)", 0))
                key_apis[name] = {"total_us": api_total_us, "count": count, "avg_us": avg_us}

            model_exec = key_apis.get("ModelExecute") or key_apis.get("aclmdlExecute")
            memcpy = key_apis.get("aclrtMemcpy")

            if model_exec:
                exec_per_iter = model_exec["total_us"] / N / 1000
                print(f"  ModelExecute:  {model_exec['total_us']/1000:.1f} ms total,"
                      f" {model_exec['count']} 次,"
                      f" {exec_per_iter:.2f} ms/iter (设备端 wall-clock)")

            if memcpy:
                memcpy_per_iter = memcpy["total_us"] / N / 1000
                print(f"  aclrtMemcpy:   {memcpy['total_us']/1000:.1f} ms total,"
                      f" {memcpy['count']} 次 ({memcpy['count']//N}/iter),"
                      f" avg {memcpy['avg_us']:.1f} μs,"
                      f" {memcpy_per_iter:.2f} ms/iter")

            kernel_per_iter = total_us / N / 1000
            if model_exec:
                gap = exec_per_iter - kernel_per_iter
                print(f"  ----")
                print(f"  kernel time:   {kernel_per_iter:.2f} ms/iter (算子执行)")
                print(f"  device wall:   {exec_per_iter:.2f} ms/iter (ModelExecute)")
                print(f"  kernel gap:    {gap:.2f} ms/iter (调度/同步开销)")
                if memcpy:
                    print(f"  memcpy 开销:   {memcpy_per_iter:.2f} ms/iter (H2D+D2D+D2H)")
                    print(f"  端到端估算:    ~{exec_per_iter + memcpy_per_iter:.2f} ms/iter"
                          f" (device wall + memcpy)")

    print("=" * 90)


if __name__ == "__main__":
    main()
