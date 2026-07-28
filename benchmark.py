"""
Benchmark: TTFT / TPOT / Peak VRAM on Ascend 310B.

Metrics:
  - TTFT (Time To First Token): Time from input submission to first output token (s)
  - TPOT (Time Per Output Token): Average time per decode token (ms)
  - Peak VRAM: Maximum NPU memory usage during inference (MB)

Output:
  - Console formatted table
  - CSV file at result/benchmark_result.csv

Usage:
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
  python benchmark.py --input_lengths 512 1024 2048 --decode_tokens 128 --num_rounds 1
"""

import os
import sys
import time
import gc
import csv
import json
import subprocess
import argparse
import threading
from datetime import datetime
import psutil
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from config import InferenceConfig
from utils.inference import Inference


# ======================== Memory Management ========================

def clear_memory():
    """Clear host memory: Python GC + kernel page cache."""
    gc.collect()
    gc.collect()
    gc.collect()
    try:
        subprocess.run(
            ["bash", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
            timeout=5, capture_output=True
        )
    except Exception:
        pass
    time.sleep(0.5)


def get_swap_mb():
    return psutil.swap_memory().used / (1024**2)


# ======================== NPU VRAM Monitoring ========================

def get_npu_memory_mb():
    """Read current NPU memory usage via npu-smi (MB)."""
    try:
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "/" in line and "310B" not in line and "Version" not in line and "Hugepages" not in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "/":
                        try:
                            used = int(parts[i - 1])
                            total = int(parts[i + 1])
                            if total > 1000:
                                return used
                        except (ValueError, IndexError):
                            pass
    except Exception:
        pass
    return -1


class VRAMMonitor:
    """Background thread that polls NPU memory and tracks peak usage."""

    def __init__(self, interval_s=0.3):
        self.interval = interval_s
        self.peak_mb = 0
        self._running = False
        self._thread = None

    def start(self):
        self.peak_mb = get_npu_memory_mb()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            mem = get_npu_memory_mb()
            if mem > self.peak_mb:
                self.peak_mb = mem
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        mem = get_npu_memory_mb()
        if mem > self.peak_mb:
            self.peak_mb = mem
        return self.peak_mb


# ======================== Benchmark Core ========================

def build_prompt(target_tokens, tokenizer):
    """Build a prompt that tokenizes to approximately target_tokens length."""
    base = (
        "Please write a very long and detailed story about a programmer who "
        "discovers an ancient computer in a cave. The computer contains the "
        "secrets of an advanced civilization. Describe every detail of their "
        "journey, the technology they find, and how it changes the world. "
        "Include dialogue, descriptions of places, and technical details. "
    )
    repeated = base * (target_tokens // 30 + 1)
    tokens = tokenizer([repeated], return_tensors="np")["input_ids"]
    actual_len = tokens.shape[1]
    if actual_len > target_tokens:
        words = repeated.split()
        ratio = target_tokens / actual_len
        word_count = int(len(words) * ratio)
        repeated = " ".join(words[:word_count])
    return repeated


def run_single_test(infer_engine, prompt, decode_tokens, vram_monitor, verbose=False):
    """Run one inference pass and measure TTFT, TPOT, Peak VRAM."""
    if verbose:
        print(f"\n    {'─'*60}")
        print(f"    [INPUT] ({len(prompt)} chars):")
        if len(prompt) > 300:
            print(f"    {prompt[:150]}")
            print(f"    ... (省略 {len(prompt)-250} chars) ...")
            print(f"    {prompt[-100:]}")
        else:
            print(f"    {prompt}")
        print(f"    {'─'*60}")

    vram_monitor.start()
    swap_before = get_swap_mb()

    ttft = 0
    token_count = 0
    decode_start = None
    output_text = ""

    for (new_text, ftl, ds, ts) in infer_engine.stream_predict(
        prompt, history=[], do_speed_test=True, max_new_tokens=decode_tokens
    ):
        output_text += new_text
        if token_count == 0:
            ttft = ftl
            decode_start = time.time()
        token_count += 1

    if decode_start and token_count > 1:
        decode_duration_s = time.time() - decode_start
        tpot_ms = (decode_duration_s / (token_count - 1)) * 1000
    else:
        tpot_ms = 0

    peak_vram = vram_monitor.stop()
    swap_after = get_swap_mb()

    if verbose:
        print(f"    [OUTPUT] ({token_count} tokens):")
        if len(output_text) > 500:
            print(f"    {output_text[:250]}")
            print(f"    ... (省略) ...")
            print(f"    {output_text[-150:]}")
        else:
            print(f"    {output_text}")
        print(f"    {'─'*60}")

    return {
        "ttft_s": ttft,
        "tpot_ms": tpot_ms,
        "decode_speed": (1000.0 / tpot_ms) if tpot_ms > 0 else 0,
        "peak_vram_mb": peak_vram,
        "output_tokens": token_count,
        "output_text": output_text,
        "swap_before_mb": swap_before,
        "swap_after_mb": swap_after,
        "swap_delta_mb": swap_after - swap_before,
    }


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: TTFT / TPOT / Peak VRAM on Ascend 310B")
    parser.add_argument("--om_model_path", type=str,
                        default="./output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_rectified.om",
                        help="Path to compiled .om model file")
    parser.add_argument("--hf_model_dir", type=str,
                        default="/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="HuggingFace model dir (for tokenizer)")
    parser.add_argument("--input_lengths", type=int, nargs="+", default=[512, 1024, 2048],
                        help="Input prompt lengths in tokens to compare")
    parser.add_argument("--decode_tokens", type=int, default=128,
                        help="Max number of tokens to decode per test")
    parser.add_argument("--num_rounds", type=int, default=1,
                        help="Repetitions per input_length")
    parser.add_argument("--num_warmup", type=int, default=0,
                        help="Warmup inferences (0=disabled)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print input prompt and output text for each test")
    parser.add_argument("--output_dir", type=str, default="./result",
                        help="Directory to save result CSV and JSON")
    args = parser.parse_args()

    kv_cache_length = 4096
    os.makedirs(args.output_dir, exist_ok=True)

    config = InferenceConfig(
        hf_model_dir=args.hf_model_dir,
        om_model_path=args.om_model_path,
        onnx_model_path="",
        session_type="acl",
        max_batch=1,
        max_input_length=kv_cache_length - 1,
        max_output_length=kv_cache_length,
        kv_cache_length=kv_cache_length,
        max_prefill_length=1,
        dtype="float16",
        temperature=0.6,
        system_prompt="",
    )

    # ---- Header ----
    print("=" * 72)
    print("BENCHMARK: DeepSeek-R1-Distill-Qwen-1.5B on Ascend 310B (NPU)")
    print("=" * 72)
    print(f"  Model          : {os.path.basename(args.om_model_path)}")
    print(f"  KV Cache       : {kv_cache_length}")
    print(f"  Input lengths  : {args.input_lengths} tokens")
    print(f"  Decode tokens  : {args.decode_tokens}")
    print(f"  Rounds         : {args.num_rounds}")
    print(f"  Warmup         : {args.num_warmup}")
    print(f"  Temperature    : 0.6")
    print(f"  RAM            : {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  Swap used      : {get_swap_mb():.0f} MB")
    print(f"  NPU VRAM       : {get_npu_memory_mb()} MB / 11577 MB")
    print("=" * 72)

    # ---- Load ----
    print("\nClearing memory...")
    clear_memory()
    print("Loading model...")
    t0 = time.time()
    infer_engine = Inference(config)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s | NPU VRAM={get_npu_memory_mb()}MB | Swap={get_swap_mb():.0f}MB")

    # ---- Optional Warmup ----
    if args.num_warmup > 0:
        print(f"\nWarmup ({args.num_warmup} rounds)...")
        for _ in range(args.num_warmup):
            for _ in infer_engine.stream_predict(
                "Hello", history=[], do_speed_test=False, max_new_tokens=8
            ):
                pass
            infer_engine.session.reset()
        print("Warmup done.")

    # ---- Run Benchmarks ----
    all_results = []

    for input_len in args.input_lengths:
        print(f"\n{'━'*72}")
        print(f"  INPUT: {input_len} tokens → Decode max {args.decode_tokens} tokens")
        print(f"{'━'*72}")

        prompt = build_prompt(input_len, infer_engine.tokenizer)
        actual_input_tokens = infer_engine.tokenizer(
            [prompt], return_tensors="np"
        )["input_ids"].shape[1]
        print(f"  Actual input tokens: {actual_input_tokens}")

        for rd in range(args.num_rounds):
            clear_memory()
            infer_engine.session.reset()

            vram_mon = VRAMMonitor(interval_s=0.3)
            result = run_single_test(
                infer_engine, prompt, args.decode_tokens, vram_mon, verbose=args.verbose
            )
            result["input_len"] = actual_input_tokens
            result["target_input_len"] = input_len
            result["round"] = rd + 1

            swap_flag = " [SWAP!]" if abs(result["swap_delta_mb"]) > 100 else ""
            print(f"  Round {rd+1}: TTFT={result['ttft_s']:.3f}s | "
                  f"TPOT={result['tpot_ms']:.1f}ms ({result['decode_speed']:.2f} tok/s) | "
                  f"Peak VRAM={result['peak_vram_mb']}MB | "
                  f"Output={result['output_tokens']} tokens{swap_flag}")

            all_results.append(result)

    # ---- Summary Table (for display) ----
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    # Group by input_len
    header = (f"{'Input(tok)':<12} {'Output(tok)':<12} {'TTFT(s)':<10} "
              f"{'TPOT(ms)':<10} {'Speed(tok/s)':<13} {'Peak VRAM(MB)':<15}")
    print(f"\n{header}")
    print("─" * 72)

    summary_rows = []
    for input_len in args.input_lengths:
        group = [r for r in all_results if r["target_input_len"] == input_len]
        ttfts = [r["ttft_s"] for r in group]
        tpots = [r["tpot_ms"] for r in group if r["tpot_ms"] > 0]
        speeds = [r["decode_speed"] for r in group if r["decode_speed"] > 0]
        vrams = [r["peak_vram_mb"] for r in group if r["peak_vram_mb"] > 0]
        out_toks = [r["output_tokens"] for r in group]

        row = {
            "input_tokens": input_len,
            "output_tokens": int(np.mean(out_toks)) if out_toks else 0,
            "ttft_s": np.mean(ttfts) if ttfts else 0,
            "tpot_ms": np.mean(tpots) if tpots else 0,
            "decode_speed": np.mean(speeds) if speeds else 0,
            "peak_vram_mb": max(vrams) if vrams else 0,
        }
        summary_rows.append(row)

        print(f"{row['input_tokens']:<12} {row['output_tokens']:<12} "
              f"{row['ttft_s']:<10.3f} {row['tpot_ms']:<10.1f} "
              f"{row['decode_speed']:<13.2f} {row['peak_vram_mb']:<15}")

    print("─" * 72)

    # ---- Save CSV ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"benchmark_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "input_tokens", "output_tokens",
            "ttft_s", "tpot_ms", "decode_speed_tok_s", "peak_vram_mb",
            "swap_delta_mb", "round"
        ])
        for r in all_results:
            writer.writerow([
                os.path.basename(args.om_model_path),
                r["input_len"],
                r["output_tokens"],
                f"{r['ttft_s']:.4f}",
                f"{r['tpot_ms']:.2f}",
                f"{r['decode_speed']:.2f}",
                r["peak_vram_mb"],
                f"{r['swap_delta_mb']:.1f}",
                r["round"],
            ])
    print(f"\nCSV saved: {csv_path}")

    # ---- Save JSON (for programmatic use) ----
    json_path = os.path.join(args.output_dir, f"benchmark_{timestamp}.json")
    json_data = {
        "model": os.path.basename(args.om_model_path),
        "device": "Ascend 310B1",
        "kv_cache_length": kv_cache_length,
        "max_prefill_length": 1,
        "temperature": 0.6,
        "timestamp": timestamp,
        "summary": summary_rows,
        "detail": [{k: v for k, v in r.items() if k != "output_text"} for r in all_results],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_path}")

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
