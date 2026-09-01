#!/bin/bash
# scripts/cleanup_memory.sh — 测试前后清理内存，防止 OOM 导致 SSH 卡死
#
# 用法:
#   sudo bash scripts/cleanup_memory.sh          # 完整清理
#   sudo bash scripts/cleanup_memory.sh --check  # 仅检查，不清理
#
# 功能:
#   1. 杀死所有占用 NPU 的 Python 推理进程
#   2. 清理 Linux 页缓存 / dentries / inodes
#   3. 确认 SWAP 使用量为 0，否则强制回收
#   4. 打印清理前后的内存状态对比

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DEVICE_ID=0
NPU_MEM_LIMIT_MB=7885   # 7.7 GB: 11577 - 3300(系统固定) - 512(安全余量)
CHECK_ONLY=false

if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=true
fi

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

get_npu_mem_used() {
    npu-smi info 2>/dev/null \
        | grep "Memory-Usage" -A4 \
        | grep -oP '\d+\s*/\s*\d+' \
        | tail -1 \
        | awk -F'/' '{gsub(/ /,"",$1); print $1}'
}

get_swap_used_mb() {
    free -m | awk '/^Swap:/ {print $3}'
}

get_sys_mem_available_mb() {
    free -m | awk '/^Mem:/ {print $7}'
}

print_status() {
    local label="$1"
    echo ""
    log_info "===== 内存状态 [$label] ====="

    local npu_used
    npu_used=$(get_npu_mem_used)
    local npu_total
    npu_total=$(npu-smi info 2>/dev/null \
        | grep "Memory-Usage" -A4 \
        | grep -oP '\d+\s*/\s*\d+' \
        | tail -1 \
        | awk -F'/' '{gsub(/ /,"",$2); print $2}')

    local sys_avail
    sys_avail=$(get_sys_mem_available_mb)
    local swap_used
    swap_used=$(get_swap_used_mb)

    echo "  NPU 内存:    ${npu_used:-?} / ${npu_total:-?} MB"
    echo "  系统可用:    ${sys_avail} MB"
    echo "  SWAP 使用:   ${swap_used} MB"
    echo "  NPU 限制:    ${NPU_MEM_LIMIT_MB} MB"

    if [[ -n "$npu_used" ]] && (( npu_used > NPU_MEM_LIMIT_MB )); then
        log_error "NPU 内存 ${npu_used} MB 超出 ${NPU_MEM_LIMIT_MB} MB 限制!"
        return 1
    fi

    if (( swap_used > 0 )); then
        log_error "SWAP 使用 ${swap_used} MB > 0，违反零 SWAP 约束!"
        return 1
    fi

    log_info "资源状态正常"
    return 0
}

kill_npu_python_procs() {
    log_info "查找并终止占用 NPU 的 Python 进程..."

    local pids
    pids=$(fuser /dev/davinci${DEVICE_ID} 2>/dev/null | tr -s ' ' '\n' | sort -u || true)

    if [[ -z "$pids" ]]; then
        log_info "无进程占用 /dev/davinci${DEVICE_ID}"
        return
    fi

    for pid in $pids; do
        local cmdline
        cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' || echo "unknown")
        log_warn "终止进程 PID=$pid: $cmdline"
        kill -9 "$pid" 2>/dev/null || true
    done

    sleep 2
    log_info "已清理 NPU 进程"
}

flush_page_cache() {
    log_info "清理 Linux 页缓存 (drop_caches=3)..."
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || {
        log_warn "drop_caches 需要 root 权限，跳过"
        return
    }
    log_info "页缓存已清理"
}

reclaim_swap() {
    local swap_used
    swap_used=$(get_swap_used_mb)

    if (( swap_used == 0 )); then
        log_info "SWAP 使用量为 0，无需回收"
        return
    fi

    log_warn "SWAP 使用 ${swap_used} MB，正在回收 (swapoff/swapon)..."
    swapoff -a 2>/dev/null && swapon -a 2>/dev/null || {
        log_error "SWAP 回收失败，请手动处理"
        return 1
    }
    log_info "SWAP 已回收"
}

set_swappiness_zero() {
    local current
    current=$(cat /proc/sys/vm/swappiness)
    if (( current != 0 )); then
        log_info "设置 swappiness: ${current} -> 0（防止使用 SWAP）"
        echo 0 > /proc/sys/vm/swappiness 2>/dev/null || {
            log_warn "设置 swappiness 需要 root 权限，跳过"
            return
        }
    else
        log_info "swappiness 已为 0"
    fi
}

# ===== 主流程 =====

echo "========================================"
echo " 内存清理工具 — Atlas 200I A2 (310B1)"
echo "========================================"

print_status "当前状态" || true

if $CHECK_ONLY; then
    exit 0
fi

echo ""
log_info "开始清理..."

kill_npu_python_procs   || log_warn "NPU 进程清理异常，继续执行"
flush_page_cache        || log_warn "页缓存清理异常，继续执行"
reclaim_swap            || log_warn "SWAP 回收异常，继续执行"
set_swappiness_zero     || log_warn "swappiness 设置异常，继续执行"

print_status "清理后" || true
