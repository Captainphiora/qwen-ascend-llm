#!/usr/bin/env python3
"""
NPU / 内存 / 交换区 诊断 + 一键清理工具

功能:
  1. 诊断模式 (默认): 显示 NPU显存、系统内存、Swap、占用进程的详细信息
  2. 清理模式 (--clean): 执行一键清理
  3. 监控模式 (--watch): 持续监控，每隔N秒刷新

用法:
  python cleanup.py              # 只看诊断信息
  python cleanup.py --clean      # 诊断 + 一键清理
  python cleanup.py --watch 3    # 每3秒刷新一次监控
  python cleanup.py --kill-npu   # 杀掉占用NPU的进程（慎用）

清理内容:
  - Python gc.collect()
  - 杀死残留的 Python 推理进程（可选）
  - sync + drop_caches=3 (释放 page cache)
  - swapoff/swapon (强制清空交换区，需要足够空闲 RAM)
"""

import os
import sys
import gc
import time
import signal
import argparse
import subprocess
import re
from pathlib import Path


# ======================== 颜色输出 ========================

class Color:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def red(s): return f"{Color.RED}{s}{Color.END}"
def green(s): return f"{Color.GREEN}{s}{Color.END}"
def yellow(s): return f"{Color.YELLOW}{s}{Color.END}"
def blue(s): return f"{Color.BLUE}{s}{Color.END}"
def bold(s): return f"{Color.BOLD}{s}{Color.END}"


# ======================== 信息采集 ========================

def get_mem_info():
    """获取系统内存信息 (MB)"""
    import psutil
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "ram_total_mb": vm.total / (1024**2),
        "ram_used_mb": vm.used / (1024**2),
        "ram_available_mb": vm.available / (1024**2),
        "ram_percent": vm.percent,
        "ram_buffers_mb": vm.buffers / (1024**2) if hasattr(vm, 'buffers') else 0,
        "ram_cached_mb": vm.cached / (1024**2) if hasattr(vm, 'cached') else 0,
        "swap_total_mb": sw.total / (1024**2),
        "swap_used_mb": sw.used / (1024**2),
        "swap_free_mb": sw.free / (1024**2),
        "swap_percent": sw.percent,
    }


def get_npu_info():
    """获取 NPU 显存信息"""
    info = {"vram_used_mb": -1, "vram_total_mb": -1, "aicore_pct": 0, "temp_c": 0, "power_w": 0}
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5)
        lines = result.stdout.splitlines()
        for idx, line in enumerate(lines):
            # Line with 310B has power and temp: "| 0  310B1  | OK  | 8.5  39  0 / 0 |"
            if "310B" in line:
                parts = line.replace("|", " ").split()
                # Find numbers after health status
                nums = re.findall(r'[\d.]+', line)
                # Format: 0, 310, 1(from "310B1"), power, temp, hp_used, hp_total
                if len(nums) >= 5:
                    info["power_w"] = float(nums[3])
                    info["temp_c"] = int(float(nums[4]))
            # Line with memory: "| 0  0  | NA  | 0  2651 / 11577 |"
            if "/" in line and "310B" not in line and "Version" not in line and "Hugepages" not in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "/" and i > 0 and i < len(parts) - 1:
                        try:
                            used = int(parts[i-1])
                            total = int(parts[i+1])
                            if total > 1000:
                                info["vram_used_mb"] = used
                                info["vram_total_mb"] = total
                        except ValueError:
                            pass
    except Exception:
        pass
    return info


def get_top_memory_processes(n=10):
    """获取占用内存最多的进程"""
    import psutil
    procs = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
        try:
            mem = proc.info['memory_info']
            if mem:
                rss_mb = mem.rss / (1024**2)
                swap_mb = getattr(mem, 'swap', 0) / (1024**2) if hasattr(mem, 'swap') else 0
                cmdline = ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else proc.info['name']
                procs.append({
                    "pid": proc.info['pid'],
                    "name": proc.info['name'],
                    "rss_mb": rss_mb,
                    "swap_mb": swap_mb,
                    "cmdline": cmdline[:80],
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    procs.sort(key=lambda x: x["rss_mb"], reverse=True)
    return procs[:n]


def get_swap_processes():
    """获取使用 swap 的进程（从 /proc/[pid]/smaps_rollup 读取）"""
    swap_procs = []
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            smaps = pid_dir / "smaps_rollup"
            if smaps.exists():
                content = smaps.read_text()
                for line in content.splitlines():
                    if line.startswith("Swap:"):
                        swap_kb = int(line.split()[1])
                        if swap_kb > 0:
                            # Get process name
                            comm = (pid_dir / "comm").read_text().strip()
                            cmdline = (pid_dir / "cmdline").read_bytes().decode(
                                errors='replace').replace('\x00', ' ')[:80]
                            swap_procs.append({
                                "pid": int(pid_dir.name),
                                "name": comm,
                                "swap_mb": swap_kb / 1024,
                                "cmdline": cmdline,
                            })
                        break
        except (PermissionError, FileNotFoundError, OSError):
            pass
    swap_procs.sort(key=lambda x: x["swap_mb"], reverse=True)
    return swap_procs


def find_npu_processes():
    """查找可能占用 NPU 的进程"""
    import psutil
    npu_procs = []
    keywords = ["acl", "npu", "ascend", "davinci", "hiai", "om_model", "cli_chat", "benchmark", "api.py"]
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            name = proc.info['name'] or ''
            if any(kw in cmdline.lower() or kw in name.lower() for kw in keywords):
                if proc.pid != os.getpid():  # 排除自己
                    rss = proc.info['memory_info'].rss / (1024**2) if proc.info['memory_info'] else 0
                    npu_procs.append({
                        "pid": proc.pid,
                        "name": name,
                        "rss_mb": rss,
                        "cmdline": cmdline[:100],
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return npu_procs


# ======================== 诊断输出 ========================

def print_diagnosis():
    """打印完整诊断信息"""
    print(bold("\n" + "=" * 72))
    print(bold("  系统资源诊断"))
    print(bold("=" * 72))

    # NPU
    npu = get_npu_info()
    print(bold("\n[NPU 显存]"))
    if npu["vram_total_mb"] > 0:
        pct = npu["vram_used_mb"] / npu["vram_total_mb"] * 100
        color_fn = red if pct > 80 else (yellow if pct > 50 else green)
        vram_str = f"{npu['vram_used_mb']}MB"
        print(f"  Used / Total : {color_fn(vram_str)} / {npu['vram_total_mb']}MB ({pct:.1f}%)")
        print(f"  Temp         : {npu['temp_c']}°C")
        print(f"  Power        : {npu['power_w']}W")
    else:
        print(red("  无法读取 NPU 信息"))

    # RAM
    mem = get_mem_info()
    print(bold("\n[系统内存]"))
    ram_color = red if mem["ram_percent"] > 85 else (yellow if mem["ram_percent"] > 70 else green)
    ram_str = f"{mem['ram_used_mb']:.0f}MB"
    print(f"  Used / Total : {ram_color(ram_str)} / {mem['ram_total_mb']:.0f}MB ({mem['ram_percent']:.1f}%)")
    print(f"  Available    : {mem['ram_available_mb']:.0f}MB")
    print(f"  Buffers      : {mem['ram_buffers_mb']:.0f}MB")
    print(f"  Cached       : {mem['ram_cached_mb']:.0f}MB")

    # Swap
    print(bold("\n[交换区 Swap]"))
    swap_color = red if mem["swap_percent"] > 20 else (yellow if mem["swap_percent"] > 5 else green)
    swap_used_str = f"{mem['swap_used_mb']:.0f}MB"
    print(f"  Used / Total : {swap_color(swap_used_str)} / {mem['swap_total_mb']:.0f}MB ({mem['swap_percent']:.1f}%)")

    # Swap进程详情
    swap_procs = get_swap_processes()
    if swap_procs:
        print(bold("\n  占用 Swap 的进程 (Top 5):"))
        print(f"  {'PID':<8} {'Swap(MB)':<10} {'Name':<15} {'Command'}")
        print(f"  {'─'*8} {'─'*10} {'─'*15} {'─'*40}")
        for p in swap_procs[:5]:
            print(f"  {p['pid']:<8} {p['swap_mb']:<10.1f} {p['name']:<15} {p['cmdline'][:40]}")
    else:
        print(green("  没有进程使用 Swap"))

    # Top memory processes
    print(bold("\n[内存占用 Top 10 进程]"))
    top_procs = get_top_memory_processes(10)
    print(f"  {'PID':<8} {'RSS(MB)':<10} {'Name':<15} {'Command'}")
    print(f"  {'─'*8} {'─'*10} {'─'*15} {'─'*40}")
    for p in top_procs:
        print(f"  {p['pid']:<8} {p['rss_mb']:<10.1f} {p['name']:<15} {p['cmdline'][:40]}")

    # NPU-related processes
    npu_procs = find_npu_processes()
    if npu_procs:
        print(bold("\n[疑似占用 NPU 的进程]"))
        print(f"  {'PID':<8} {'RSS(MB)':<10} {'Name':<15} {'Command'}")
        print(f"  {'─'*8} {'─'*10} {'─'*15} {'─'*40}")
        for p in npu_procs:
            print(f"  {p['pid']:<8} {p['rss_mb']:<10.1f} {p['name']:<15} {p['cmdline'][:40]}")
    else:
        print(green("\n[NPU 进程] 未发现残留的 NPU 推理进程"))

    print("")
    return mem, npu, npu_procs


# ======================== 清理操作 ========================

def do_clean(kill_npu_procs=False, force_swap_clear=False):
    """执行一键清理"""
    print(bold("\n" + "─" * 72))
    print(bold("  执行清理"))
    print(bold("─" * 72))

    # 1. Python GC
    print("\n  [1/5] Python gc.collect()...", end=" ")
    gc.collect()
    gc.collect()
    gc.collect()
    print(green("done"))

    # 2. Kill NPU processes
    if kill_npu_procs:
        npu_procs = find_npu_processes()
        if npu_procs:
            print(f"\n  [2/5] 杀死 NPU 相关进程 ({len(npu_procs)} 个)...")
            for p in npu_procs:
                try:
                    print(f"        kill {p['pid']} ({p['name']})...", end=" ")
                    os.kill(p["pid"], signal.SIGTERM)
                    print(green("SIGTERM sent"))
                except ProcessLookupError:
                    print(yellow("already gone"))
                except PermissionError:
                    print(red("permission denied"))
            time.sleep(2)
            # Force kill survivors
            for p in npu_procs:
                try:
                    os.kill(p["pid"], 0)  # Check if alive
                    os.kill(p["pid"], signal.SIGKILL)
                    print(f"        SIGKILL {p['pid']}")
                except (ProcessLookupError, PermissionError):
                    pass
        else:
            print("\n  [2/5] 无 NPU 残留进程，跳过")
    else:
        print("\n  [2/5] 跳过杀进程 (使用 --kill-npu 启用)")

    # 3. Drop page cache
    print("\n  [3/5] sync + drop_caches=3...", end=" ")
    try:
        subprocess.run(["sync"], timeout=10)
        result = subprocess.run(
            ["bash", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(green("done"))
        else:
            print(yellow(f"failed: {result.stderr.strip()}"))
    except Exception as e:
        print(red(f"error: {e}"))

    # 4. Clear swap (swapoff + swapon)
    if force_swap_clear:
        import psutil
        swap_used = psutil.swap_memory().used / (1024**2)
        ram_avail = psutil.virtual_memory().available / (1024**2)

        if swap_used < 1:
            print("\n  [4/5] Swap 已空，跳过")
        elif ram_avail < swap_used + 500:
            print(f"\n  [4/5] " + red(f"RAM 不足以容纳 Swap 内容 (需要 {swap_used:.0f}MB, 可用 {ram_avail:.0f}MB)"))
            print(f"        跳过 swapoff（强制执行可能导致 OOM Kill）")
        else:
            print(f"\n  [4/5] swapoff + swapon (清空 {swap_used:.0f}MB swap)...", end=" ")
            try:
                r = subprocess.run(["swapoff", "-a"], capture_output=True, text=True, timeout=60)
                if r.returncode == 0:
                    subprocess.run(["swapon", "-a"], capture_output=True, timeout=10)
                    print(green("done"))
                else:
                    print(red(f"swapoff failed: {r.stderr.strip()}"))
            except subprocess.TimeoutExpired:
                print(red("timeout (swap too large?)"))
                subprocess.run(["swapon", "-a"], capture_output=True, timeout=10)
            except Exception as e:
                print(red(f"error: {e}"))
    else:
        import psutil
        swap_used = psutil.swap_memory().used / (1024**2)
        if swap_used > 10:
            print(f"\n  [4/5] Swap 有 {swap_used:.0f}MB 占用，使用 --clean-swap 强制清空")
        else:
            print(f"\n  [4/5] Swap 占用很低 ({swap_used:.0f}MB)，跳过")

    # 5. Set swappiness to low value
    print("\n  [5/5] 设置 vm.swappiness=10...", end=" ")
    try:
        result = subprocess.run(
            ["sysctl", "-w", "vm.swappiness=10"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(green("done (减少后续 swap 使用概率)"))
        else:
            print(yellow(f"failed: {result.stderr.strip()}"))
    except Exception as e:
        print(red(f"error: {e}"))

    print(bold("\n  清理完成!"))


# ======================== 监控模式 ========================

def watch_mode(interval):
    """持续监控模式"""
    print(f"监控模式: 每 {interval} 秒刷新 (Ctrl+C 退出)\n")
    try:
        while True:
            os.system("clear")
            npu = get_npu_info()
            mem = get_mem_info()

            print(bold(f"{'='*60}"))
            print(bold(f"  实时监控 (每 {interval}s)    按 Ctrl+C 退出"))
            print(bold(f"{'='*60}"))

            # One-line NPU
            if npu["vram_total_mb"] > 0:
                pct = npu["vram_used_mb"] / npu["vram_total_mb"] * 100
                bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                print(f"\n  NPU VRAM : [{bar}] {npu['vram_used_mb']:>5}MB / {npu['vram_total_mb']}MB ({pct:.1f}%)")
                print(f"  NPU Temp : {npu['temp_c']}°C | Power: {npu['power_w']}W")

            # One-line RAM
            bar = "█" * int(mem["ram_percent"] // 5) + "░" * (20 - int(mem["ram_percent"] // 5))
            print(f"\n  RAM      : [{bar}] {mem['ram_used_mb']:>5.0f}MB / {mem['ram_total_mb']:.0f}MB ({mem['ram_percent']:.1f}%)")
            print(f"  Available: {mem['ram_available_mb']:.0f}MB | Cached: {mem['ram_cached_mb']:.0f}MB")

            # One-line Swap
            if mem["swap_total_mb"] > 0:
                spct = mem["swap_percent"]
                bar = "█" * int(spct // 5) + "░" * (20 - int(spct // 5))
                color_fn = red if spct > 20 else (yellow if spct > 5 else green)
                swap_str = f"{mem['swap_used_mb']:.0f}MB"
                print(f"\n  Swap     : [{bar}] {color_fn(swap_str)} / {mem['swap_total_mb']:.0f}MB ({spct:.1f}%)")
                if spct > 5:
                    print(red("  ⚠ Swap 占用中 — 推理性能可能下降！"))

            # NPU processes
            npu_procs = find_npu_processes()
            if npu_procs:
                print(f"\n  NPU进程: {len(npu_procs)} 个")
                for p in npu_procs[:3]:
                    print(f"    PID={p['pid']} {p['name']} ({p['rss_mb']:.0f}MB)")

            print(f"\n  {time.strftime('%H:%M:%S')}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n监控结束")


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(
        description="NPU/内存/Swap 诊断与一键清理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cleanup.py                    # 诊断（只看不动）
  python cleanup.py --clean            # 诊断 + 清理 page cache + 降低 swappiness
  python cleanup.py --clean --clean-swap          # 同上 + 强制清空 swap
  python cleanup.py --clean --kill-npu            # 同上 + 杀掉 NPU 残留进程
  python cleanup.py --clean --kill-npu --clean-swap  # 全部清理
  python cleanup.py --watch 2          # 每2秒监控
        """)
    parser.add_argument("--clean", action="store_true",
                        help="执行清理操作 (gc + drop_caches + 降低swappiness)")
    parser.add_argument("--kill-npu", action="store_true",
                        help="杀死占用NPU的残留进程 (慎用)")
    parser.add_argument("--clean-swap", action="store_true",
                        help="强制清空交换区 (swapoff/swapon，需要足够空闲RAM)")
    parser.add_argument("--watch", type=float, metavar="SEC", default=0,
                        help="监控模式: 每N秒刷新一次")

    args = parser.parse_args()

    # Watch mode
    if args.watch > 0:
        watch_mode(args.watch)
        return

    # Diagnosis
    mem, npu, npu_procs = print_diagnosis()

    # Clean
    if args.clean:
        do_clean(kill_npu_procs=args.kill_npu, force_swap_clear=args.clean_swap)
        # Show after
        print(bold("\n" + "─" * 72))
        print(bold("  清理后状态"))
        print(bold("─" * 72))
        time.sleep(1)
        print_diagnosis()
    else:
        # Suggestions
        suggestions = []
        if mem["swap_used_mb"] > 10:
            suggestions.append("python cleanup.py --clean --clean-swap  # 清空 swap")
        if npu_procs:
            suggestions.append("python cleanup.py --clean --kill-npu    # 杀 NPU 残留进程")
        if mem["ram_cached_mb"] > 500:
            suggestions.append("python cleanup.py --clean               # 释放 page cache")

        if suggestions:
            print(bold("\n[建议操作]"))
            for s in suggestions:
                print(f"  {s}")
        print("")


if __name__ == "__main__":
    main()
