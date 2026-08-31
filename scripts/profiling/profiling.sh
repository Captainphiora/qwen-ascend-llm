#!/bin/bash

# ============================================================
# 昇腾 NPU Profiling 一键脚本（采集 + 导出 + 分析）
# ============================================================
# 使用方法:
#   1. 修改下方【参数配置区】中的参数
#   2. 直接运行: bash profiling.sh
# ============================================================

# ============================================================
# 【参数配置区】直接在这里修改参数，然后运行脚本
# ============================================================

# ---------- 必填：要 profiling 的 Python 脚本路径 ----------
SCRIPT="inference.py"

# ---------- 可选：传给 Python 脚本的参数 ----------
# 示例: "--model qwen2 --batch 1 --seq_len 128"
SCRIPT_ARGS=""

# ---------- Profiling 输出目录 ----------
OUTPUT_DIR="./profiling_output"

# ---------- CANN 安装路径 ----------
CANN_HOME="/usr/local/Ascend/cann-9.0.0"

# ---------- msprof 采集选项 ----------

# task-time: 是否采集算子 task 级别耗时
#   候选: on / off
#   说明: 开启后可在 op_summary 中看到每个算子的精确执行时间
TASK_TIME="on"

# ai-core: 是否采集 AI Core (Cube单元) 的 profiling 数据
#   候选: on / off
#   说明: 开启后可获得 mac_ratio, mte2_ratio 等流水线占比数据
AI_CORE="on"

# aic-metrics: AI Core 采集的指标组
#   候选: PipeUtilization        - 各流水线利用率 (mac/mte1/mte2/mte3/vec/scalar ratio)
#          ArithmeticUtilization  - 算术单元利用率
#          Memory                 - 内存访问统计
#          MemoryL0               - L0 Buffer 访问统计
#          MemoryUB               - Unified Buffer 访问统计
#          ResourceConflictRatio  - 资源冲突比例
#          L2Cache                - L2 Cache 命中率
#          PipelineExecuteUtilization - 流水线执行利用率
#   说明: PipeUtilization 是最常用的，可判断计算/访存瓶颈
AIC_METRICS="PipeUtilization"

# aic-mode: AI Core 采集模式
#   候选: task-based   - 按算子 task 粒度采集（精确，推荐）
#          sample-based - 按固定频率采样（开销小，但可能遗漏短算子）
AIC_MODE="task-based"

# aicpu: 是否采集 AI CPU 算子数据
#   候选: on / off
#   说明: 开启后可检测是否有算子 fallback 到 AI CPU 执行
AICPU="on"

# runtime-api: 是否采集 Host 侧 Runtime API 调用耗时
#   候选: on / off
#   说明: 开启后可看到 aclmdlExecute/MemCopy/Launch 等 Host CPU 耗时
RUNTIME_API="on"

# hccl: 是否采集集合通信 (多卡) 数据
#   候选: on / off
#   说明: 单卡推理无需开启；多卡分布式场景开启可分析通信瓶颈
HCCL="off"

# msproftx: 是否采集用户自定义打点 (msproftx/mstx) 数据
#   候选: on / off
#   说明: 需要在代码中主动调用 mstx 打点 API 才有数据
MSPROFTX="off"

# l2: 是否采集 L2 Cache 数据
#   候选: on / off
#   说明: 开启后可分析 L2 Cache 命中率，对访存优化有帮助
L2="off"

# ---------- 导出选项 ----------

# 导出类型
#   候选: summary  - 只导出统计 CSV
#          timeline - 只导出 Timeline JSON
#          all      - 两者都导出（推荐）
EXPORT_TYPE="all"

# ============================================================
# 【以下为执行逻辑，一般不需要修改】
# ============================================================

set -e

TOOL_DIR="${CANN_HOME}/tools/profiler/profiler_tool/analysis"

if [ ! -f "$SCRIPT" ]; then
    echo "[ERROR] 脚本不存在: $SCRIPT"
    echo "        请修改脚本顶部的 SCRIPT 变量"
    exit 1
fi

if [ ! -d "$CANN_HOME" ]; then
    echo "[ERROR] 未找到 CANN: $CANN_HOME"
    echo "        请修改脚本顶部的 CANN_HOME 变量"
    exit 1
fi

source "${CANN_HOME}/set_env.sh"

# ============================================================
# 第一步：采集 Profiling 数据
# ============================================================
echo ""
echo "============================================================"
echo " 第一步：采集 Profiling 数据"
echo "============================================================"
echo "[INFO] 目标脚本: python3 $SCRIPT $SCRIPT_ARGS"
echo "[INFO] 输出目录: $OUTPUT_DIR"
echo "[INFO] 采集配置: ai-core=$AI_CORE, metrics=$AIC_METRICS, mode=$AIC_MODE"
echo ""

mkdir -p "$OUTPUT_DIR"

msprof \
    --output="$OUTPUT_DIR" \
    --task-time="$TASK_TIME" \
    --ai-core="$AI_CORE" \
    --aic-metrics="$AIC_METRICS" \
    --aic-mode="$AIC_MODE" \
    --aicpu="$AICPU" \
    --runtime-api="$RUNTIME_API" \
    --hccl="$HCCL" \
    --msproftx="$MSPROFTX" \
    --l2="$L2" \
    python3 "$SCRIPT" $SCRIPT_ARGS

echo ""
echo "[INFO] 采集完成"

# ============================================================
# 第二步：找到生成的 PROF 目录
# ============================================================
PROF_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "PROF_*" | sort | tail -1)

if [ -z "$PROF_DIR" ]; then
    echo "[ERROR] 未找到 PROF 目录，采集可能失败"
    exit 1
fi

PROF_DIR=$(realpath "$PROF_DIR")
echo "[INFO] PROF 目录: $PROF_DIR"

# ============================================================
# 第三步：导出 Summary 和 Timeline
# ============================================================
echo ""
echo "============================================================"
echo " 第二步：导出分析报告"
echo "============================================================"

run_msprof_export() {
    local type="$1"
    echo "[INFO] 正在导出 ${type} ..."
    cd "$TOOL_DIR" && python3 -c "
from msinterface.msprof_entrance import MsprofEntrance
import sys
sys.argv = ['msprof', 'export', '${type}', '-dir', '${PROF_DIR}']
MsprofEntrance().main()
"
}

case "$EXPORT_TYPE" in
    summary)
        run_msprof_export summary
        ;;
    timeline)
        run_msprof_export timeline
        ;;
    all)
        run_msprof_export summary
        run_msprof_export timeline
        ;;
esac

OUTPUT_CSV_DIR="${PROF_DIR}/mindstudio_profiler_output"
echo ""
echo "[INFO] 导出完成，文件位于: $OUTPUT_CSV_DIR"

# ============================================================
# 第四步：自动分析并打印报告
# ============================================================
echo ""
echo "============================================================"
echo " 第三步：性能分析报告"
echo "============================================================"

python3 << PYEOF
import csv
import os
import glob

output_dir = "${OUTPUT_CSV_DIR}"

stat_files = sorted(glob.glob(os.path.join(output_dir, "op_statistic_*.csv")))
summary_files = sorted(glob.glob(os.path.join(output_dir, "op_summary_*.csv")))
step_files = sorted(glob.glob(os.path.join(output_dir, "step_trace_*.csv")))

if not stat_files:
    print("[WARN] 未找到 op_statistic 文件")
    exit(0)

# ---- 算子类型统计 ----
print("\n【算子类型耗时 Top10】")
print(f"  {'类型':<25} {'核心类型':<18} {'总耗时(ms)':>10} {'占比':>8}")
print("  " + "-" * 65)
with open(stat_files[-1], 'r') as f:
    reader = csv.DictReader(f)
    rows = sorted(reader, key=lambda r: -float(r.get('Total Time(us)', 0)))
    total_us = sum(float(r.get('Total Time(us)', 0)) for r in rows)
    for r in rows[:10]:
        t = float(r.get('Total Time(us)', 0))
        print(f"  {r['OP Type']:<25} {r['Core Type']:<18} {t/1000:>10.1f} {t/total_us*100:>7.1f}%")

# ---- AI_CPU 检测 ----
print("\n【AI_CPU Fallback 检测】")
aicpu_rows = [r for r in rows if r.get('Core Type') == 'AI_CPU']
if aicpu_rows:
    print("  [!] 发现以下算子 fallback 到 AI_CPU:")
    for r in aicpu_rows:
        print(f"    {r['OP Type']:<20} 次数={r['Count']:<5} 总耗时={float(r['Total Time(us)'])/1000:.1f}ms")
else:
    print("  [OK] 无 AI_CPU fallback，所有算子均在 NPU 上执行")

# ---- 瓶颈分析 ----
if summary_files:
    print("\n【瓶颈分析 (MatMul 算子的 mte2_ratio vs mac_ratio)】")
    mte2_vals = []
    mac_vals = []
    with open(summary_files[-1], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['OP Type'] == 'BatchMatMulV2':
                try:
                    mte2 = float(row['mte2_ratio'])
                    mac = float(row['mac_ratio'])
                    mte2_vals.append(mte2)
                    mac_vals.append(mac)
                except:
                    pass
    if mte2_vals:
        avg_mte2 = sum(mte2_vals) / len(mte2_vals)
        avg_mac = sum(mac_vals) / len(mac_vals)
        print(f"  MatMul 平均 mte2_ratio (访存占比): {avg_mte2:.3f}")
        print(f"  MatMul 平均 mac_ratio  (计算占比): {avg_mac:.3f}")
        if avg_mte2 > avg_mac * 3:
            print(f"  --> 结论: 【访存瓶颈 (Memory Bound)】")
            print(f"      mte2 >> mac，NPU 大部分时间在等数据从内存搬运到片上")
        elif avg_mac > avg_mte2 * 3:
            print(f"  --> 结论: 【计算瓶颈 (Compute Bound)】")
            print(f"      mac >> mte2，搬运已完成但计算单元忙不过来")
        else:
            print(f"  --> 结论: 【计算与访存接近平衡】")

# ---- Step Trace ----
if step_files:
    print("\n【Iteration 耗时统计】")
    iters = []
    with open(step_files[-1], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iters.append(float(row['Iteration Time(us)']))
            except:
                pass
    if iters:
        steady = iters[3:] if len(iters) > 5 else iters
        avg_iter = sum(steady) / len(steady)
        print(f"  总 iteration 数: {len(iters)}")
        print(f"  稳态平均耗时:    {avg_iter/1000:.1f} ms/iter")
        print(f"  对应吞吐:        {1e6/avg_iter:.2f} tokens/s")

print("\n【输出文件】")
for f in sorted(os.listdir(output_dir)):
    print(f"  {f}")
print(f"\n  Timeline JSON 可用 chrome://tracing 或 https://ui.perfetto.dev 可视化")
print("")
PYEOF

echo "============================================================"
echo " 完成！"
echo "============================================================"
