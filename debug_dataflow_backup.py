"""
调试脚本：模拟 ACL 推理的完整数据流动过程
用伪造数据模拟 prefill + decode 全流程，打印每一步的数据状态

用法: python debug_dataflow.py --seq_len 14 --kv_cache_length 16 --max_prefill_length 4 --decode_steps 4
"""

import numpy as np
import math
import argparse


# ============================================================
# 模拟配置
# ============================================================
class FakeConfig:
    def __init__(self, kv_cache_length, max_prefill_length):
        self.max_batch = 1
        self.kv_cache_length = kv_cache_length
        self.max_prefill_length = max_prefill_length
        self.num_hidden_layers = 2       # 简化：2层
        self.num_key_value_heads = 2     # 简化：2个KV头
        self.per_head_dim = 4            # 简化：每头4维
        self.vocab_size = 10             # 简化：词表大小10
        self.past_key_value_shape = [
            self.max_batch,
            self.kv_cache_length,
            self.num_hidden_layers * 2 * self.num_key_value_heads,  # 8
            self.per_head_dim,  # 4
        ]
        self.half_past_key_value_shape = list(self.past_key_value_shape)
        self.half_past_key_value_shape[1] = self.kv_cache_length // 2


# ============================================================
# 模拟 Device 内存（用 numpy 数组模拟 NPU 显存）
# ============================================================
class FakeDeviceMemory:
    """模拟 Device (NPU) 侧的一块内存"""
    def __init__(self, name, shape, dtype=np.float16):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data = np.zeros(shape, dtype=dtype)
        self.size_bytes = self.data.nbytes

    def memset_zero(self):
        self.data[:] = 0

    def __repr__(self):
        return f"DeviceMem[{self.name}] shape={self.shape} bytes={self.size_bytes}"


# ============================================================
# 模拟 ACLModel（对应 utils/engine.py 的 ACLModel）
# ============================================================
class FakeACLModel:
    def __init__(self, config: FakeConfig):
        self.config = config
        self.max_batch = config.max_batch
        self.kv_cache_length = config.kv_cache_length
        self.max_prefill_length = config.max_prefill_length
        self.past_key_value_shape = config.past_key_value_shape
        self.half_past_key_value_shape = config.half_past_key_value_shape

        self.input_pos = 0
        self.real_kv_size = 0

        # 模拟 Device 内存分配 (对应 engine.py allocate_memory)
        # inputs[0]: input_ids, inputs[1]: mask, inputs[2]: pos_ids, inputs[3]: kv_cache
        self.kv_cache_device = FakeDeviceMemory(
            "kv_cache (inputs[3])",
            shape=tuple(config.past_key_value_shape),
        )
        # outputs[0]: logits, outputs[1]: new_kv_cache
        self.logits_device = FakeDeviceMemory(
            "logits (outputs[0])",
            shape=(config.max_batch, config.max_prefill_length, config.vocab_size),
        )

        print("=" * 70)
        print("[ 初始化 ] 模拟 Device 内存分配 (对应 acl.rt.malloc)")
        print(f"  {self.kv_cache_device}")
        print(f"  {self.logits_device}")
        print(f"  KV Cache 全长 shape: {config.past_key_value_shape}")
        print(f"  KV Cache 半长 shape: {config.half_past_key_value_shape}")
        print("=" * 70)

    def reset(self):
        """对应 engine.py:150-160 ACLModel.reset()"""
        print("\n" + "─" * 70)
        print("[ reset ] acl.rt.memset(inputs[3], 0) — Device 端 KV Cache 置零")
        print(f"  操作位置: Device (NPU显存)")
        print(f"  操作内容: 将 {self.kv_cache_device.size_bytes} bytes 全部写0")
        self.input_pos = 0
        self.real_kv_size = 0
        self.kv_cache_device.memset_zero()
        print(f"  reset 后: input_pos=0, real_kv_size=0")
        print("─" * 70)

    def get_inputs(self, seq_len: int):
        """对应 engine.py:117-148，在 Host 端构造 mask 和 pos_id"""
        temp_seq_len = self.real_kv_size + seq_len
        if self.max_prefill_length > 1 and temp_seq_len <= self.kv_cache_length // 2:
            temp_kv_size = self.kv_cache_length // 2
            kv_mode = "半长"
        else:
            temp_kv_size = self.kv_cache_length
            kv_mode = "全长"

        mask = np.ones((1, temp_kv_size + seq_len), dtype=np.int64)
        mask[:, self.real_kv_size: temp_kv_size] = 0

        pos_id = np.arange(
            self.input_pos,
            self.input_pos + seq_len,
            dtype=np.int64
        ).reshape(1, -1)

        print(f"\n  [ get_inputs ] 在 Host 端构造 mask 和 pos_id")
        print(f"    执行位置: Host (CPU)")
        print(f"    KV模式: {kv_mode} (temp_kv_size={temp_kv_size})")
        print(f"    real_kv_size={self.real_kv_size}, input_pos={self.input_pos}, seq_len={seq_len}")
        print(f"    mask shape: {mask.shape} = (1, kv_size({temp_kv_size}) + seq_len({seq_len}))")
        print(f"    mask value: {mask[0].tolist()}")
        self._explain_mask(mask[0], temp_kv_size, seq_len)
        print(f"    pos_id: {pos_id[0].tolist()}")

        return mask, pos_id, temp_kv_size, kv_mode

    def _explain_mask(self, mask, temp_kv_size, seq_len):
        """可视化 mask 的含义"""
        parts = []
        # KV区域
        kv_valid = self.real_kv_size
        kv_empty = temp_kv_size - self.real_kv_size
        if kv_valid > 0:
            parts.append(f"[1×{kv_valid}=有效KV]")
        if kv_empty > 0:
            parts.append(f"[0×{kv_empty}=空位/masked]")
        parts.append(f"[1×{seq_len}=当前输入]")
        print(f"    mask 含义: {' '.join(parts)}")

    def inference(self, input_data_list, seq_length, is_dynamic, is_prefill):
        """
        对应 engine.py:333-470 ACLModel.inference()
        模拟完整的推理过程
        """
        input_ids, mask, pos_ids = input_data_list

        print(f"\n  [ inference ] is_prefill={is_prefill}, is_dynamic={is_dynamic}")

        # 步骤1: Host → Device 拷贝
        print(f"    ┌─ 步骤1: Host → Device 拷贝 (acl.rt.memcpy HOST_TO_DEVICE)")
        print(f"    │  input_ids: {input_ids[0].tolist()} → Device inputs[0]")
        print(f"    │  mask:      ({mask.shape[1]} elements) → Device inputs[1]")
        print(f"    │  pos_ids:   {pos_ids[0].tolist()} → Device inputs[2]")
        print(f"    │  注意: inputs[3](KV Cache) 不拷贝，始终留在 Device")
        print(f"    │")

        # 步骤2: 设置动态shape
        if is_dynamic:
            if (self.real_kv_size + seq_length) > self.kv_cache_length // 2:
                kv_shape_used = self.past_key_value_shape
                label = "全长"
            else:
                kv_shape_used = self.half_past_key_value_shape
                label = "半长"

            print(f"    ├─ 步骤2: 设置动态 shape (acl.mdl.set_input_dynamic_dims)")
            print(f"    │  input_ids shape: (1, {seq_length})")
            print(f"    │  mask shape:      (1, {mask.shape[1]})")
            print(f"    │  pos_ids shape:   (1, {seq_length})")
            print(f"    │  kv_cache shape:  {kv_shape_used} ({label})")
            print(f"    │  执行位置: Host 调用 API，通知 Device 使用此档位")
            print(f"    │")

        # 步骤3: 模型执行
        print(f"    ├─ 步骤3: acl.mdl.execute() — 模型前向计算")
        print(f"    │  执行位置: Device (NPU)")
        print(f"    │  计算: {seq_length} 个 token 对 {self.real_kv_size} 个已有 KV 做 attention")
        print(f"    │  输出: logits shape=(1,{seq_length},{self.config.vocab_size}), new_kv shape=(1,{seq_length},8,4)")

        # 模拟输出：用随机数模拟 new_kv_cache
        new_kv = np.random.randn(1, seq_length, 8, 4).astype(np.float16)
        logits = np.random.randn(1, seq_length, self.config.vocab_size).astype(np.float16)

        print(f"    │")

        # 步骤4: 更新 KV Cache (Device → Device)
        print(f"    ├─ 步骤4: update_kv_cache (acl.rt.memcpy DEVICE_TO_DEVICE)")
        old_real_kv = self.real_kv_size
        self._update_kv_cache(seq_length, new_kv)
        print(f"    │")

        # 步骤5: 是否拷回 logits
        if not is_prefill:
            print(f"    └─ 步骤5: 拷回 logits (acl.rt.memcpy DEVICE_TO_HOST)")
            print(f"       执行位置: Device → Host")
            print(f"       大小: (1, {seq_length}, {self.config.vocab_size}) × 2 bytes = {seq_length * self.config.vocab_size * 2} bytes")
            print(f"       ★ 非 prefill，需要 logits 来采样下一个 token")
            return logits
        else:
            print(f"    └─ 步骤5: 跳过 logits 拷贝!")
            print(f"       ★ prefill 阶段，只需 KV Cache 副产物，省掉 D2H 传输")
            return None

    def _update_kv_cache(self, seq_len, new_kv):
        """对应 engine.py:162-190"""
        self.input_pos = self.real_kv_size + seq_len

        if seq_len + self.real_kv_size > self.kv_cache_length:
            actual_write = self.kv_cache_length - self.real_kv_size
            print(f"    │  ⚠ KV Cache 即将溢出! 只能写入 {actual_write} 个位置")
            if actual_write <= 0:
                print(f"    │  ⚠ KV Cache 已满! 新 KV 被丢弃!")
                print(f"    │  real_kv_size={self.real_kv_size} (不变), input_pos={self.input_pos}")
                return
            seq_len = actual_write

        write_start = self.real_kv_size
        write_end = self.real_kv_size + seq_len
        print(f"    │  执行位置: Device (NPU显存内搬运)")
        print(f"    │  操作: outputs[1][0:seq] → inputs[3][{write_start}:{write_end}]")
        print(f"    │  含义: 将 {seq_len} 个新 token 的 KV 写入 Cache 的位置 {write_start}~{write_end - 1}")

        # 模拟写入
        self.kv_cache_device.data[:, write_start:write_end] = new_kv[:, :seq_len]
        self.real_kv_size += seq_len
        print(f"    │  更新后: real_kv_size={self.real_kv_size}, input_pos={self.input_pos}")


# ============================================================
# 模拟 AclSession（对应 utils/session.py 的 AclSession）
# ============================================================
class FakeAclSession:
    def __init__(self, config: FakeConfig):
        self.config = config
        self.model = FakeACLModel(config)
        self.max_batch = config.max_batch
        self.max_prefill_length = config.max_prefill_length
        self.prefill_log2_number = int(math.log2(self.max_prefill_length))
        self.prefill_log2_list = [2 ** i for i in range(self.prefill_log2_number, -1, -1)]
        self.run_times = 0

        print(f"\n[ Session 配置 ]")
        print(f"  max_prefill_length={self.max_prefill_length}")
        print(f"  prefill_log2_list={self.prefill_log2_list} (可用的分块大小)")
        print(f"  kv_cache_length={config.kv_cache_length}")

    def reset(self):
        self.model.reset()

    def decompose_number(self, n, start_index=0):
        """对应 session.py:181-196，将 n 分解为 2 的幂次之和"""
        if n == 0:
            return []
        for i in range(start_index, self.prefill_log2_number + 1):
            power = self.prefill_log2_list[i]
            if power <= n:
                return [power] + self.decompose_number(n - power, i)
        return []

    def run(self, input_ids: np.ndarray):
        """对应 session.py:198-232"""
        seq_len = input_ids.shape[-1]
        is_dynamic = bool(self.max_prefill_length > 1)

        print(f"\n{'=' * 70}")
        print(f"[ AclSession.run ] input_ids shape={input_ids.shape}, seq_len={seq_len}")
        print(f"  is_dynamic={is_dynamic}")

        if is_dynamic:
            seq_list = self.decompose_number(seq_len)
            print(f"  分解 {seq_len} → {seq_list} (2的幂次贪心分解)")
            print(f"  将分 {len(seq_list)} 块执行, 最后一块 is_prefill=False")

            logits = None
            start_i = 0
            for ii, seq in enumerate(seq_list):
                end_i = start_i + seq
                is_prefill = (ii != len(seq_list) - 1)

                print(f"\n{'─' * 70}")
                print(f"▶ 块 {ii}/{len(seq_list)-1}: input_ids[:, {start_i}:{end_i}], "
                      f"seq={seq}, is_prefill={is_prefill}")

                logits = self.run_some(
                    input_ids[:, start_i:end_i],
                    seq, is_dynamic, is_prefill
                )
                start_i += seq
            return logits
        else:
            # 静态推理：逐 token
            logits = None
            for i in range(seq_len):
                is_prefill = (i != seq_len - 1)
                logits = self.run_some(input_ids[:, i:i+1], 1, False, is_prefill)
            return logits

    def run_some(self, input_ids, seq_length=1, is_dynamic=False, is_prefill=False):
        """对应 session.py:234-254"""
        self.run_times += seq_length
        mask, pos_ids, temp_kv_size, kv_mode = self.model.get_inputs(seq_length)

        logits = self.model.inference(
            [input_ids, mask, pos_ids], seq_length, is_dynamic, is_prefill
        )

        if not is_prefill:
            return logits.reshape(self.max_batch, seq_length, -1)
        else:
            return None


# ============================================================
# 模拟采样
# ============================================================
def fake_sample(logits):
    """模拟 greedy 采样：取 argmax"""
    next_token = np.argmax(logits[0, -1, :])
    return np.array([[next_token]], dtype=np.int64)


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="调试 ACL 推理数据流")
    parser.add_argument("--seq_len", type=int, default=14, help="输入 prompt 长度")
    parser.add_argument("--kv_cache_length", type=int, default=16, help="KV Cache 总长度")
    parser.add_argument("--max_prefill_length", type=int, default=4, help="最大 prefill 分块(需为2的幂)")
    parser.add_argument("--decode_steps", type=int, default=4, help="decode 阶段生成的 token 数")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          ACL 推理数据流调试脚本 (伪造数据)                          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n参数: seq_len={args.seq_len}, kv_cache_length={args.kv_cache_length}, "
          f"max_prefill_length={args.max_prefill_length}, decode_steps={args.decode_steps}")

    config = FakeConfig(args.kv_cache_length, args.max_prefill_length)
    session = FakeAclSession(config)

    # 伪造输入 token ids
    input_ids = np.arange(100, 100 + args.seq_len, dtype=np.int64).reshape(1, -1)
    print(f"\n[ 输入 ] input_ids = {input_ids[0].tolist()}")
    print(f"  (模拟 tokenizer 输出, 值为 100~{99 + args.seq_len})")

    # ============ Prefill 阶段 ============
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                     PREFILL 阶段                                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    session.reset()
    logits = session.run(input_ids)

    print(f"\n[ Prefill 完成 ]")
    print(f"  返回 logits shape: {logits.shape}")
    print(f"  取 logits[0][-1:] 用于采样第一个生成 token")

    next_token = fake_sample(logits)
    print(f"  采样结果: token={next_token[0, 0]}")
    print(f"  当前 KV Cache 状态: real_kv_size={session.model.real_kv_size}/{config.kv_cache_length}")

    # ============ Decode 阶段 ============
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                     DECODE 阶段                                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    generated_tokens = [next_token[0, 0]]

    for step in range(args.decode_steps):
        print(f"\n{'━' * 70}")
        print(f"  DECODE STEP {step + 1}/{args.decode_steps}")
        print(f"  输入: token={next_token[0, 0]}, shape={next_token.shape}")
        print(f"  KV Cache 状态: {session.model.real_kv_size}/{config.kv_cache_length} 已用")

        logits = session.run(next_token)

        if logits is not None:
            next_token = fake_sample(logits)
            generated_tokens.append(next_token[0, 0])
            print(f"\n  采样结果: token={next_token[0, 0]}")
        else:
            print(f"\n  ⚠ logits 为 None (不应该在 decode 阶段发生)")
            break

        kv_used = session.model.real_kv_size
        kv_total = config.kv_cache_length
        bar_len = 30
        filled = int(bar_len * kv_used / kv_total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  KV Cache: [{bar}] {kv_used}/{kv_total}")

    # ============ 最终总结 ============
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                     总结                                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  输入 tokens:  {input_ids[0].tolist()}")
    print(f"  生成 tokens:  {generated_tokens}")
    print(f"  总 token 数:  {args.seq_len + len(generated_tokens)}")
    print(f"  KV Cache 使用: {session.model.real_kv_size}/{config.kv_cache_length}")
    if session.model.real_kv_size >= config.kv_cache_length:
        print(f"  ⚠ KV Cache 已满! 后续生成的 token 的 KV 无法被记录")
    print(f"\n  Prefill 阶段调用模型 {len(session.decompose_number(args.seq_len))} 次")
    print(f"  Decode  阶段调用模型 {args.decode_steps} 次")
    print(f"  总计模型执行次数: {len(session.decompose_number(args.seq_len)) + args.decode_steps}")


if __name__ == "__main__":
    main()
