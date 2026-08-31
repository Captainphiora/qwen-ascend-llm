#!/bin/bash
# ============================================================
# 性能对比脚本：顺序运行所有版本的 benchmark
#
# 用法:
#   bash run_benchmark_comparison.sh
# ============================================================

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo " 性能对比测试"
echo "============================================================"
echo ""

RESULT_DIR="./benchmark_results"
mkdir -p "$RESULT_DIR"

# 运行所有已有的 benchmark 脚本
for bench_script in scripts/bench_v*.sh; do
    if [ -f "$bench_script" ]; then
        echo ""
        echo ">>> 运行: $bench_script"
        echo ""
        bash "$bench_script"
        echo ""
    fi
done

echo ""
echo "============================================================"
echo " 所有测试完成，结果在: $RESULT_DIR/"
echo "============================================================"
ls -la "$RESULT_DIR/"
