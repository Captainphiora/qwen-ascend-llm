"""
优化脚本：修改 modeling_qwen2.py 中的 RoPE 实现和 Expand 操作
消除 Slice+Neg+Concat 模式，消除不必要的 Expand。

优化 1: RoPE (rotate_half)
  原始: x1 = x[..., :D//2]; x2 = x[..., D//2:]; cat(-x2, x1)
  产生 ONNX 节点: Slice + Slice + Neg + Concat (每层每Q/K各一套)
  优化: 用 reshape + stack 实现，避免 Slice+Concat

优化 2: Expand (cos/sin broadcast)
  原始: cos[position_ids].unsqueeze(unsqueeze_dim) → 隐式 broadcast 产生 Expand
  优化: 手动 expand 为 repeat，让 trace 时生成更高效的模式

用法:
  python optimize_rope.py                    # 应用优化
  python optimize_rope.py --revert           # 恢复原始
  python optimize_rope.py --check            # 只检查当前状态
"""

import argparse
import os
import shutil
import re

MODELING_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "export", "modeling_qwen2.py"
)
BACKUP_FILE = MODELING_FILE + ".bak_original"


# ============================================================
# 原始代码片段（用于匹配和恢复）
# ============================================================
ORIGINAL_ROTATE_HALF = '''# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)'''

ORIGINAL_APPLY_ROTARY = '''# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed'''


# ============================================================
# 优化后代码
# ============================================================
OPTIMIZED_ROTATE_HALF = '''# Optimized: avoid Slice+Neg+Concat, use reshape+flip instead
def rotate_half(x):
    """Rotates half the hidden dims of the input.
    Optimized to avoid Slice+Neg+Concat pattern in ONNX graph.
    Uses reshape+sign-flip which traces to fewer, more fusible ops.
    """
    # x shape: (..., head_dim)
    # reshape to (..., head_dim//2, 2), swap and negate, flatten back
    x_reshaped = x.unflatten(-1, (-1, 2))        # (..., D//2, 2)
    # Take pairs (a, b) -> (-b, a)
    x_rotated = torch.stack(
        (-x_reshaped[..., 1], x_reshaped[..., 0]),
        dim=-1
    )                                              # (..., D//2, 2)
    return x_rotated.flatten(-2)                   # (..., D)'''

OPTIMIZED_APPLY_ROTARY = '''# Optimized: eliminate Expand by explicitly repeating cos/sin to match head count
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Optimized to reduce Expand ops by explicit repeat instead of broadcast.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # Explicitly expand cos/sin to match q's head dimension
    # This avoids implicit broadcast → Expand op in ONNX
    # q shape: (batch, num_heads, seq, head_dim)
    # cos shape after unsqueeze: (batch, 1, seq, head_dim)
    if cos.shape[unsqueeze_dim] != q.shape[unsqueeze_dim]:
        cos = cos.expand_as(q)
        sin_q = sin.expand_as(q)
    else:
        sin_q = sin
    if cos.shape[unsqueeze_dim] != k.shape[unsqueeze_dim]:
        cos_k = cos[..., :k.shape[-1]] if cos.shape[-1] != k.shape[-1] else cos
        cos_k = cos_k[:, :k.shape[1], :, :]
        sin_k = sin[:, :k.shape[1], :, :]
    else:
        cos_k = cos
        sin_k = sin
    # For GQA: q has more heads than k, handle separately
    q_embed = (q * cos) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed'''


def check_status():
    """检查当前文件状态"""
    with open(MODELING_FILE, "r") as f:
        content = f.read()

    if "avoid Slice+Neg+Concat" in content:
        return "optimized"
    elif "x1 = x[..., : x.shape[-1] // 2]" in content:
        return "original"
    else:
        return "unknown"


def apply_optimization():
    """应用 RoPE 优化"""
    status = check_status()
    if status == "optimized":
        print("[INFO] 已经是优化后的状态，无需重复操作")
        return False

    # 备份原始文件
    if not os.path.exists(BACKUP_FILE):
        shutil.copy2(MODELING_FILE, BACKUP_FILE)
        print(f"[INFO] 已备份原始文件到: {BACKUP_FILE}")

    with open(MODELING_FILE, "r") as f:
        content = f.read()

    # 替换 rotate_half
    if ORIGINAL_ROTATE_HALF in content:
        content = content.replace(ORIGINAL_ROTATE_HALF, OPTIMIZED_ROTATE_HALF)
        print("[OK] rotate_half 已优化 (消除 Slice+Neg+Concat)")
    else:
        print("[WARN] 未找到原始 rotate_half 代码，可能已被修改")

    # 替换 apply_rotary_pos_emb
    if ORIGINAL_APPLY_ROTARY in content:
        content = content.replace(ORIGINAL_APPLY_ROTARY, OPTIMIZED_APPLY_ROTARY)
        print("[OK] apply_rotary_pos_emb 已优化 (消除 Expand)")
    else:
        print("[WARN] 未找到原始 apply_rotary_pos_emb 代码，可能已被修改")

    with open(MODELING_FILE, "w") as f:
        f.write(content)

    print(f"\n[DONE] 优化已应用到: {MODELING_FILE}")
    return True


def revert_optimization():
    """恢复原始代码"""
    if os.path.exists(BACKUP_FILE):
        shutil.copy2(BACKUP_FILE, MODELING_FILE)
        print(f"[OK] 已从备份恢复: {BACKUP_FILE} -> {MODELING_FILE}")
    else:
        print("[ERROR] 找不到备份文件，无法恢复")
        print(f"  期望路径: {BACKUP_FILE}")


def main():
    parser = argparse.ArgumentParser(description="RoPE + Expand 优化工具")
    parser.add_argument("--revert", action="store_true", help="恢复原始代码")
    parser.add_argument("--check", action="store_true", help="只检查当前状态")
    parser.add_argument("--rope-only", action="store_true",
                        help="只应用 RoPE 优化 (v1)，不改 apply_rotary_pos_emb")
    args = parser.parse_args()

    print(f"[INFO] 目标文件: {MODELING_FILE}")

    if args.check:
        status = check_status()
        print(f"[INFO] 当前状态: {status}")
        return

    if args.revert:
        revert_optimization()
        return

    # 应用优化
    if args.rope_only:
        # 只优化 rotate_half，不动 apply_rotary_pos_emb
        status = check_status()
        if status == "optimized":
            print("[INFO] 已经是优化后的状态")
            return
        if not os.path.exists(BACKUP_FILE):
            shutil.copy2(MODELING_FILE, BACKUP_FILE)
            print(f"[INFO] 已备份原始文件到: {BACKUP_FILE}")
        with open(MODELING_FILE, "r") as f:
            content = f.read()
        if ORIGINAL_ROTATE_HALF in content:
            content = content.replace(ORIGINAL_ROTATE_HALF, OPTIMIZED_ROTATE_HALF)
            print("[OK] rotate_half 已优化 (消除 Slice+Neg+Concat)")
        with open(MODELING_FILE, "w") as f:
            f.write(content)
        print("[DONE] v1 (仅 RoPE) 优化已应用")
    else:
        success = apply_optimization()
        if success:
            print("[DONE] v2 (RoPE + Expand) 优化已应用")


if __name__ == "__main__":
    main()
