#!/usr/bin/env python3
"""Calculate theoretical peak throughput for LLM decode on Ascend NPU.

Reads model config (HuggingFace config.json) and profiling CSV to compute:
  1. Per-token weight traffic (from architecture)
  2. Per-token total traffic (from profiling)
  3. Per-token FLOPs
  4. Roofline analysis & bottleneck determination
  5. Theoretical peak throughput under memory-bound / compute-bound

Usage:
    python scripts/calc_theoretical_throughput.py
    python scripts/calc_theoretical_throughput.py --profiling_csv <path> --config_json <path>
    python scripts/calc_theoretical_throughput.py --peak_bw 204.8  # for 310P
"""

import argparse
import csv
import json
import os
import sys


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
    if 'FLOAT16' in d:
        return 2
    if 'FLOAT' in d:
        return 4
    if 'INT8' in d:
        return 1
    if 'INT32' in d or 'INT64' in d:
        return 4
    return 2


def load_model_config(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    return {
        'hidden_size': cfg['hidden_size'],
        'intermediate_size': cfg['intermediate_size'],
        'num_hidden_layers': cfg['num_hidden_layers'],
        'num_attention_heads': cfg['num_attention_heads'],
        'num_kv_heads': cfg.get('num_key_value_heads', cfg['num_attention_heads']),
        'head_dim': cfg['hidden_size'] // cfg['num_attention_heads'],
        'vocab_size': cfg['vocab_size'],
    }


def calc_weight_bytes(cfg, dtype_size=2):
    """Per-token weight traffic: all weight matrices loaded once.
    
    Args:
        cfg: model config dict
        dtype_size: bytes per element (2=FP16, 1=INT8 for quantized weights)
    """
    h = cfg['hidden_size']
    inter = cfg['intermediate_size']
    n_layers = cfg['num_hidden_layers']
    kv_heads = cfg['num_kv_heads']
    head_dim = cfg['head_dim']
    vocab = cfg['vocab_size']

    qkv_size = h * (h + 2 * kv_heads * head_dim)
    o_size = h * h
    gate_up_size = h * inter * 2
    down_size = inter * h
    ln_size = h * 4

    per_layer = (qkv_size + o_size + gate_up_size + down_size + ln_size) * dtype_size
    lm_head = h * vocab * dtype_size
    final_ln = h * 2 * dtype_size

    total = n_layers * per_layer + lm_head + final_ln
    return {
        'per_layer': per_layer,
        'lm_head': lm_head,
        'total': total,
        'attention_per_layer': (qkv_size + o_size) * dtype_size,
        'mlp_per_layer': (gate_up_size + down_size) * dtype_size,
    }


def calc_flops(cfg):
    """Per-token FLOPs (batch=1, seq_len=1 decode)."""
    h = cfg['hidden_size']
    inter = cfg['intermediate_size']
    n_layers = cfg['num_hidden_layers']
    kv_heads = cfg['num_kv_heads']
    head_dim = cfg['head_dim']
    vocab = cfg['vocab_size']

    qkv_flops = 2 * h * (h + 2 * kv_heads * head_dim)
    o_flops = 2 * h * h
    mlp_flops = 2 * h * inter * 2 + 2 * inter * h
    per_layer = qkv_flops + o_flops + mlp_flops
    lm_head = 2 * h * vocab

    total = n_layers * per_layer + lm_head
    return {
        'per_layer': per_layer,
        'lm_head': lm_head,
        'total': total,
    }


GATHER_OPS = {'GatherV2', 'Gather', 'GatherElements', 'GatherND',
              'EmbeddingLookup'}


def analyze_profiling(csv_path, num_tokens):
    """Analyze profiling CSV for total memory traffic and bottleneck type.

    For most ops: traffic = sum(input sizes) + sum(output sizes).
    For Gather-family ops: the declared input[0] is the full source tensor,
    but only the gathered slice is actually read from HBM (≈ output size).
    Corrected traffic = sum(output sizes) * 2  (read gathered + write output)
                       + sum(index input sizes)  (tiny, usually negligible).
    """
    total_mem = 0
    total_mem_raw = 0
    total_time_us = 0
    mac_weighted = 0
    mte2_weighted = 0
    total_aic_us = 0

    op_stats = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row['Task Duration(us)'])
            total_time_us += dur

            ins = row.get('Input Shapes', '')
            outs = row.get('Output Shapes', '')
            idts = row.get('Input Data Types', '')
            odts = row.get('Output Data Types', '')

            in_shapes = [parse_shape(s) for s in ins.split(';') if s.strip()]
            out_shapes = [parse_shape(s) for s in outs.split(';') if s.strip()]
            in_dtypes = [s.strip() for s in idts.split(';') if s.strip()]
            out_dtypes = [s.strip() for s in odts.split(';') if s.strip()]

            op_type = row['OP Type']

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
            total_mem += mem

            aic = float(row.get('aicore_time(us)', 0) or 0)
            mac = float(row.get('aic_mac_ratio', row.get('mac_exe_ratio', 0)) or 0)
            mte2 = float(row.get('aic_mte2_ratio', row.get('mte2_exe_ratio', 0)) or 0)
            total_aic_us += aic
            mac_weighted += mac * aic
            mte2_weighted += mte2 * aic

            op_type = row['OP Type']
            if op_type not in op_stats:
                op_stats[op_type] = {'count': 0, 'time_us': 0, 'mac_w': 0, 'mte2_w': 0, 'aic_us': 0}
            d = op_stats[op_type]
            d['count'] += 1
            d['time_us'] += dur
            d['mac_w'] += mac * aic
            d['mte2_w'] += mte2 * aic
            d['aic_us'] += aic

    avg_mac = mac_weighted / total_aic_us if total_aic_us > 0 else 0
    avg_mte2 = mte2_weighted / total_aic_us if total_aic_us > 0 else 0

    return {
        'total_mem_bytes': total_mem,
        'total_mem_raw_bytes': total_mem_raw,
        'total_time_us': total_time_us,
        'num_tokens': num_tokens,
        'per_token_mem': total_mem / num_tokens,
        'per_token_mem_raw': total_mem_raw / num_tokens,
        'per_token_time_ms': total_time_us / num_tokens / 1000,
        'eff_bw_gbs': total_mem / 1e9 / (total_time_us / 1e6),
        'eff_bw_raw_gbs': total_mem_raw / 1e9 / (total_time_us / 1e6),
        'avg_mac': avg_mac,
        'avg_mte2': avg_mte2,
        'is_memory_bound': avg_mte2 > avg_mac * 2,
        'op_stats': op_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Decode 理论峰值吞吐量计算")
    parser.add_argument("--config_json", type=str,
                        default="models/DeepSeek-R1-Distill-Qwen-1.5B/config.json",
                        help="HuggingFace model config.json path")
    parser.add_argument("--profiling_csv", type=str, default="",
                        help="Profiling op_summary CSV (auto-detect if empty)")
    parser.add_argument("--num_tokens", type=int, default=0,
                        help="Number of tokens in profiling (0=auto-detect from step_trace)")
    parser.add_argument("--step_trace_csv", type=str, default="",
                        help="step_trace CSV for auto-detecting token count")
    parser.add_argument("--peak_bw", type=float, default=51.2,
                        help="Peak memory bandwidth in GB/s (default: 51.2 for 310B1)")
    parser.add_argument("--peak_flops", type=float, default=10.0,
                        help="Peak FP16 compute in TFLOPS (default: 10.0 for 310B1)")
    parser.add_argument("--weight_dtype_size", type=int, default=2,
                        help="Weight dtype bytes: 2=FP16 (default), 1=INT8 (W8A8 quantized)")
    args = parser.parse_args()

    if not os.path.exists(args.config_json):
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models/DeepSeek-R1-Distill-Qwen-1.5B/config.json")
        if os.path.exists(alt):
            args.config_json = alt
        else:
            print(f"[ERROR] config.json not found: {args.config_json}")
            sys.exit(1)

    if not args.profiling_csv:
        candidates = []
        for root, dirs, files in os.walk("profiling_runs"):
            for f in files:
                if "op_summary" in f and f.endswith(".csv"):
                    candidates.append(os.path.join(root, f))
        if candidates:
            args.profiling_csv = sorted(candidates)[-1]
            print(f"[INFO] Auto-detected profiling CSV: {args.profiling_csv}")

    if not args.step_trace_csv and args.profiling_csv:
        d = os.path.dirname(args.profiling_csv)
        for f in os.listdir(d):
            if "step_trace" in f and f.endswith(".csv"):
                args.step_trace_csv = os.path.join(d, f)
                break

    PEAK_BW_GBS = args.peak_bw
    PEAK_BW_BS = PEAK_BW_GBS * 1e9
    PEAK_FLOPS = args.peak_flops * 1e12
    RIDGE = PEAK_FLOPS / PEAK_BW_BS

    cfg = load_model_config(args.config_json)
    weights = calc_weight_bytes(cfg, dtype_size=args.weight_dtype_size)
    # 同时算 FP16 版本用于对比（量化模型中 lm_head/非量化层仍是 FP16）
    weights_fp16 = calc_weight_bytes(cfg, dtype_size=2)
    flops = calc_flops(cfg)

    print("=" * 72)
    print("  LLM Decode 理论峰值吞吐量分析")
    print("=" * 72)
    print()

    print("【硬件规格】")
    print(f"  FP16 算力:      {args.peak_flops} TFLOPS")
    print(f"  内存带宽:       {PEAK_BW_GBS} GB/s")
    print(f"  Roofline 拐点:  {RIDGE:.1f} FLOPs/Byte")
    print()

    print("【模型架构】")
    print(f"  层数: {cfg['num_hidden_layers']}, hidden: {cfg['hidden_size']}, "
          f"heads: {cfg['num_attention_heads']}, kv_heads: {cfg['num_kv_heads']}")
    print(f"  intermediate: {cfg['intermediate_size']}, vocab: {cfg['vocab_size']}")
    print()

    dtype_label = f"INT8" if args.weight_dtype_size == 1 else f"FP16"
    print(f"【每 token 权重搬运量 ({dtype_label})】")
    n = cfg['num_hidden_layers']
    print(f"  {n}层 Attention:  {n * weights['attention_per_layer'] / 1e6:.0f} MB")
    print(f"  {n}层 MLP:        {n * weights['mlp_per_layer'] / 1e6:.0f} MB")
    print(f"  LM head:         {weights_fp16['lm_head'] / 1e6:.0f} MB (FP16, 未量化)")
    if args.weight_dtype_size != 2:
        quant_layers_total = n * (weights['attention_per_layer'] + weights['mlp_per_layer'])
        non_quant_total = weights_fp16['lm_head'] + cfg['hidden_size'] * 2 * 2
        actual_weight_total = quant_layers_total + non_quant_total
        print(f"  合计 (混合精度): {actual_weight_total / 1e9:.3f} GB")
        print(f"    量化层 ({dtype_label}): {quant_layers_total / 1e9:.3f} GB")
        print(f"    非量化层 (FP16): {non_quant_total / 1e6:.0f} MB")
        weights['total'] = actual_weight_total
    else:
        print(f"  合计:            {weights['total'] / 1e9:.3f} GB")
    print()

    print("【每 token 计算量】")
    print(f"  {n}层 Attn+MLP:   {n * flops['per_layer'] / 1e9:.2f} GFLOPs")
    print(f"  LM head:         {flops['lm_head'] / 1e9:.2f} GFLOPs")
    print(f"  合计:            {flops['total'] / 1e9:.3f} GFLOPs")
    print(f"  (= 2 x params，batch=1 每个权重参与 1 次 multiply-add)")
    print()

    oi = flops['total'] / weights['total']
    print("【Roofline 判定】")
    print(f"  OI = {flops['total']/1e9:.3f}G / {weights['total']/1e9:.3f}G = {oi:.2f} FLOPs/Byte")
    print(f"  拐点 = {RIDGE:.1f} FLOPs/Byte")
    if oi < RIDGE:
        print(f"  OI ({oi:.1f}) << 拐点 ({RIDGE:.0f}) -> 访存瓶颈 (Memory-Bound)")
    else:
        print(f"  OI ({oi:.1f}) >= 拐点 ({RIDGE:.0f}) -> 算力瓶颈 (Compute-Bound)")
    print()

    t_mem = weights['total'] / PEAK_BW_BS * 1000
    t_comp = flops['total'] / PEAK_FLOPS * 1000
    print("【理论峰值 (只搬权重)】")
    print(f"  访存瓶颈:  {t_mem:.1f} ms/token -> {1000/t_mem:.1f} tok/s")
    print(f"  算力瓶颈:  {t_comp:.3f} ms/token -> {1000/t_comp:.0f} tok/s")
    print()

    if args.profiling_csv and os.path.exists(args.profiling_csv):
        num_tokens = args.num_tokens
        if num_tokens == 0 and args.step_trace_csv and os.path.exists(args.step_trace_csv):
            with open(args.step_trace_csv) as f:
                num_tokens = sum(1 for _ in csv.DictReader(f))
            print(f"[INFO] Auto-detected {num_tokens} tokens from step_trace")

        if num_tokens == 0:
            print("[WARN] --num_tokens not set and step_trace not found, using 35")
            num_tokens = 35

        prof = analyze_profiling(args.profiling_csv, num_tokens)

        print("=" * 72)
        print("【Profiling 实测分析】")
        print("=" * 72)
        print()
        print(f"  Token 数:        {num_tokens}")
        print(f"  总 device time:  {prof['total_time_us']/1e6:.3f} s")
        print(f"  每 token 时延:   {prof['per_token_time_ms']:.1f} ms")
        print(f"  每 token 吞吐:   {1000/prof['per_token_time_ms']:.1f} tok/s")
        print()

        per_tok_mem_gb = prof['per_token_mem'] / 1e9
        per_tok_raw_gb = prof['per_token_mem_raw'] / 1e9
        weight_gb = weights['total'] / 1e9
        activation_gb = per_tok_mem_gb - weight_gb
        gather_correction_gb = (per_tok_raw_gb - per_tok_mem_gb)

        print(f"  每 token 搬运量 (Gather 修正): {per_tok_mem_gb:.3f} GB")
        print(f"    其中权重:         {weight_gb:.3f} GB")
        print(f"    其中中间结果:     ~{activation_gb:.2f} GB (= 总搬运 - 权重)")
        if gather_correction_gb > 0.001:
            print(f"  (修正前声明值:      {per_tok_raw_gb:.3f} GB, Gather 修正减去 {gather_correction_gb:.2f} GB)")
        print(f"  有效带宽:           {prof['eff_bw_gbs']:.1f} GB/s ({prof['eff_bw_gbs']/PEAK_BW_GBS*100:.0f}% of peak)")
        print()

        print("  瓶颈判定 (Profiling mac/mte2):")
        print(f"    avg mac_ratio:  {prof['avg_mac']:.3f}")
        print(f"    avg mte2_ratio: {prof['avg_mte2']:.3f}")
        if prof['is_memory_bound']:
            print(f"    -> 访存瓶颈 (mte2 >> mac)")
        else:
            print(f"    -> 算力瓶颈或混合 (mac >= mte2)")
        print()

        print("  算子耗时 Top-10:")
        print(f"    {'算子类型':<25s} {'次数':>6s} {'耗时ms':>9s} {'占比':>6s} {'avg_mac':>8s} {'avg_mte2':>9s}")
        print(f"    {'-'*25} {'-'*6} {'-'*9} {'-'*6} {'-'*8} {'-'*9}")
        sorted_ops = sorted(prof['op_stats'].items(), key=lambda x: -x[1]['time_us'])
        for op, d in sorted_ops[:10]:
            t_ms = d['time_us'] / 1e3
            ratio = d['time_us'] / prof['total_time_us'] * 100
            a_mac = d['mac_w'] / d['aic_us'] if d['aic_us'] > 0 else 0
            a_mte = d['mte2_w'] / d['aic_us'] if d['aic_us'] > 0 else 0
            print(f"    {op:<25s} {d['count']:>6d} {t_ms:>9.1f} {ratio:>5.1f}% {a_mac:>8.3f} {a_mte:>9.3f}")
        print()

        t_total_mem = per_tok_mem_gb / PEAK_BW_GBS * 1000
        print("【含中间结果的理论值 (Gather 修正)】")
        print(f"  每 token 总搬运 {per_tok_mem_gb:.3f} GB / {PEAK_BW_GBS} GB/s = {t_total_mem:.2f} ms -> {1000/t_total_mem:.1f} tok/s")
        print()

        print("=" * 72)
        print("【差距分析】")
        print("=" * 72)
        print()
        print(f"  (A) 理论峰值 (只搬权重):       {t_mem:.2f} ms -> {1000/t_mem:.1f} tok/s")
        print(f"  (B) 含中间结果 (带宽100%):      {t_total_mem:.2f} ms -> {1000/t_total_mem:.1f} tok/s")
        print(f"  (C) 实测:                       {prof['per_token_time_ms']:.2f} ms -> {1000/prof['per_token_time_ms']:.1f} tok/s")
        print()
        gap_ab = t_total_mem - t_mem
        gap_bc = prof['per_token_time_ms'] - t_total_mem
        print(f"  A -> B: +{gap_ab:.2f} ms  (KV cache + 中间结果搬运 ~{activation_gb:.2f} GB)")
        total_kernels = sum(d['count'] for d in prof['op_stats'].values())
        kernels_per_tok = total_kernels / num_tokens
        print(f"  B -> C: +{gap_bc:.2f} ms  (带宽利用率 {prof['eff_bw_gbs']/PEAK_BW_GBS*100:.0f}% + kernel 调度开销)")
        print(f"         (每 token {kernels_per_tok:.0f} 个 kernel，含大量 <5us 小 kernel)")
    else:
        print("[INFO] 未提供 profiling CSV，跳过实测分析")
        print("[INFO] 用法: python scripts/calc_theoretical_throughput.py --profiling_csv <path>")

    print()


if __name__ == "__main__":
    main()
