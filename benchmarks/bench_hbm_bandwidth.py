"""HBM bandwidth microbenchmark for Ascend NPU.

Tests achievable memory bandwidth with different access patterns:
1. Large contiguous copy (torch.clone) - best case
2. Large MatMul (memory-bound shape) - realistic compute pattern
3. Element-wise op (Add) - vector core bandwidth
"""
import argparse
import time
import torch
import torch_npu

def bench_copy(device, sizes_mb, warmup=5, repeat=20):
    """Contiguous memory copy: measures raw HBM read+write bandwidth."""
    print("\n=== 1. Contiguous Copy (clone) ===")
    print(f"  {'Size(MB)':>10} {'Time(ms)':>10} {'BW(GB/s)':>10} {'Note':>20}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*20}")
    for mb in sizes_mb:
        n = mb * 1024 * 1024 // 2  # FP16 = 2 bytes
        x = torch.randn(n, dtype=torch.float16, device=device)
        for _ in range(warmup):
            y = x.clone()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            y = x.clone()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) / repeat * 1000
        # read + write = 2x size
        bw = 2 * mb / 1024 / (elapsed_ms / 1000)
        note = ""
        print(f"  {mb:>10} {elapsed_ms:>10.3f} {bw:>10.1f} {note:>20}")
        del x, y

def bench_matmul(device, shapes, warmup=5, repeat=20):
    """Memory-bound MatMul: simulates weight loading pattern."""
    print("\n=== 2. MatMul (memory-bound, batch=1) ===")
    print(f"  {'Shape':>20} {'Time(ms)':>10} {'BW(GB/s)':>10} {'TFLOPS':>10} {'OI':>6}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for M, K, N in shapes:
        a = torch.randn(M, K, dtype=torch.float16, device=device)
        b = torch.randn(K, N, dtype=torch.float16, device=device)
        for _ in range(warmup):
            c = torch.mm(a, b)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            c = torch.mm(a, b)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) / repeat * 1000
        flops = 2 * M * K * N
        mem_bytes = (M*K + K*N + M*N) * 2  # FP16
        bw = mem_bytes / 1e9 / (elapsed_ms / 1e3)
        tflops = flops / 1e12 / (elapsed_ms / 1e3)
        oi = flops / mem_bytes
        shape_str = f"{M}x{K}x{N}"
        print(f"  {shape_str:>20} {elapsed_ms:>10.3f} {bw:>10.1f} {tflops:>10.2f} {oi:>6.1f}")
        del a, b, c

def bench_add(device, sizes_mb, warmup=5, repeat=20):
    """Element-wise Add: vector core bandwidth."""
    print("\n=== 3. Element-wise Add ===")
    print(f"  {'Size(MB)':>10} {'Time(ms)':>10} {'BW(GB/s)':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    for mb in sizes_mb:
        n = mb * 1024 * 1024 // 2
        a = torch.randn(n, dtype=torch.float16, device=device)
        b = torch.randn(n, dtype=torch.float16, device=device)
        for _ in range(warmup):
            c = a + b
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            c = a + b
        torch.npu.synchronize()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) / repeat * 1000
        # read a + read b + write c = 3x size
        bw = 3 * mb / 1024 / (elapsed_ms / 1000)
        print(f"  {mb:>10} {elapsed_ms:>10.3f} {bw:>10.1f}")
        del a, b, c

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()

    device = f"npu:{args.device}"
    torch.npu.set_device(args.device)
    print(f"Device: {torch.npu.get_device_properties(args.device).name}")
    print(f"HBM: {torch.npu.get_device_properties(args.device).total_memory / 1e9:.1f} GB")

    sizes = [64, 256, 1024, 2048, 4096]
    bench_copy(device, sizes)

    # Simulate decode shapes: batch=1, weight read dominates
    matmul_shapes = [
        (1, 1536, 1536),      # q/k/v/o proj size
        (1, 1536, 8960),      # gate/up proj
        (1, 8960, 1536),      # down proj
        (1, 1536, 17920),     # gate_up fused (v5)
        (1, 1536, 151936),    # lm_head
        # Larger batch for comparison
        (32, 1536, 8960),
        (128, 1536, 8960),
    ]
    bench_matmul(device, matmul_shapes)

    bench_add(device, sizes)

if __name__ == "__main__":
    main()
