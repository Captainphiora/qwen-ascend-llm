#!/bin/bash
# NPU 显存清理脚本 — Atlas 200I DK A2 (310B1 SoC)
# 用法: source cleanup_npu.sh
# 功能: 杀掉所有占用 NPU 的用户进程，释放页缓存，最大化可用内存

set -e

echo "=== NPU 显存清理开始 ==="

echo "[1] 清理前状态:"
free -m | head -2
npu-smi info 2>/dev/null | grep "Memory-Usage" || echo "    (npu-smi 无法查询内存)"

echo ""
echo "[2] 杀掉占用 NPU 的 Python 推理进程..."
KILLED=0
for pid in $(ps aux | grep -E 'python.*infer|python.*run_pa|python.*mindie|torchrun' | grep -v grep | awk '{print $2}'); do
    echo "    killing PID $pid: $(ps -p $pid -o args= 2>/dev/null | head -c 80)"
    kill "$pid" 2>/dev/null && KILLED=$((KILLED+1))
done

if [ "$KILLED" -gt 0 ]; then
    echo "    等待进程退出..."
    sleep 3
    for pid in $(ps aux | grep -E 'python.*infer|python.*run_pa|python.*mindie|torchrun' | grep -v grep | awk '{print $2}'); do
        echo "    强制杀掉残留 PID $pid"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 2
else
    echo "    没有需要清理的推理进程"
fi

echo ""
echo "[3] 杀掉 TBE 编译子进程..."
for pid in $(ps aux | grep 'forkserver' | grep -v grep | awk '{print $2}'); do
    kill -9 "$pid" 2>/dev/null || true
done

echo "[4] 释放系统页缓存..."
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "    (需要 root 权限释放页缓存)"

echo ""
echo "[5] 清理后状态:"
sleep 1
free -m | head -2
npu-smi info 2>/dev/null | grep "Memory-Usage" || echo "    (npu-smi 无法查询内存)"

echo ""
echo "=== 清理完成 ==="
