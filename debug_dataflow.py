"""
调试脚本：使用项目真实代码，伪造输入，观察完整流式推理的数据流动过程。
直接调用 InferenceConfig、Inference、AclSession、ACLModel 等项目中的真实类。

用法:
    python debug_dataflow.py

可修改下方 CONFIG 区域的参数来调整测试行为。
"""

import sys
import os
import time
import numpy as np

# ============================================================
# CONFIG: 根据你的环境修改以下参数
# ============================================================
HF_MODEL_DIR = "/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
OM_MODEL_PATH = "./output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_sim.om"
SESSION_TYPE = "acl"
DEVICE_ID = 0
KV_CACHE_LENGTH = 4096
MAX_PREFILL_LENGTH = 1      # 与你的 config 一致，1 表示逐 token prefill
MAX_INPUT_LENGTH = 4095
MAX_OUTPUT_LENGTH = 4096
MAX_BATCH = 1
DTYPE = "float16"
TORCH_DTYPE = "float16"
DEVICE_STR = "npu"

# 测试参数
TEST_PROMPT = "你好"           # 测试用的简短 prompt
MAX_NEW_TOKENS = 10            # 最多生成 10 个 token，方便观察
TEMPERATURE = 0                # greedy 采样，结果确定
SYSTEM_PROMPT = ""
# ============================================================


from config import InferenceConfig
from utils.inference import Inference


def main():
    print("=" * 70)
    print(" 调试脚本：真实代码 + 真实 NPU 推理")
    print("=" * 70)

    # 1. 创建配置
    print("\n[1] 创建 InferenceConfig ...")
    config = InferenceConfig(
        hf_model_dir=HF_MODEL_DIR,
        om_model_path=OM_MODEL_PATH,
        onnx_model_path="",
        session_type=SESSION_TYPE,
        device_id=DEVICE_ID,
        max_batch=MAX_BATCH,
        max_input_length=MAX_INPUT_LENGTH,
        max_output_length=MAX_OUTPUT_LENGTH,
        kv_cache_length=KV_CACHE_LENGTH,
        max_prefill_length=MAX_PREFILL_LENGTH,
        dtype=DTYPE,
        torch_dtype=TORCH_DTYPE,
        device_str=DEVICE_STR,
        temperature=TEMPERATURE,
        sampling_method="greedy",
        sampling_value=0.95,
        system_prompt=SYSTEM_PROMPT,
    )
    print(f"    session_type={config.session_type}")
    print(f"    kv_cache_length={config.kv_cache_length}")
    print(f"    max_prefill_length={config.max_prefill_length}")
    print(f"    past_key_value_shape={config.past_key_value_shape}")
    print(f"    num_hidden_layers={config.num_hidden_layers}")
    print(f"    num_key_value_heads={config.num_key_value_heads}")
    print(f"    per_head_dim={config.per_head_dim}")
    print(f"    vocab_size={config.vocab_size}")

    # 2. 创建 Inference 实例（会加载 tokenizer + 创建 AclSession + 加载 OM 模型）
    print("\n[2] 创建 Inference 实例（加载 tokenizer + OM 模型到 NPU）...")
    t0 = time.time()
    infer_engine = Inference(config)
    print(f"    加载耗时: {time.time() - t0:.2f}s")

    # 3. 手动执行 tokenize，观察输入
    print("\n[3] Tokenize 测试 prompt ...")
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": TEST_PROMPT})

    text = infer_engine.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"    原始 prompt: {TEST_PROMPT!r}")
    print(f"    chat_template 后:\n    {text!r}")

    input_ids = infer_engine.tokenizer(
        [text], return_tensors="np"
    )["input_ids"].astype(np.int64).reshape(1, -1)
    print(f"    input_ids shape: {input_ids.shape}")
    print(f"    input_ids: {input_ids[0].tolist()}")
    print(f"    对应 tokens: {[infer_engine.tokenizer.decode([t]) for t in input_ids[0].tolist()]}")

    input_length = input_ids.shape[1]
    print(f"    输入长度: {input_length} tokens")

    # 4. 手动走一遍 stream_predict 的核心逻辑，加入详细打印
    print("\n[4] 开始推理（手动执行 stream_predict 核心逻辑）...")
    print("─" * 70)

    session = infer_engine.session
    max_output_len = min(MAX_OUTPUT_LENGTH - input_length, MAX_NEW_TOKENS)

    print(f"    max_output_len={max_output_len}")
    print(f"    采样方式: greedy (temperature=0)")

    ids_list = []
    current_input_ids = input_ids

    for i in range(max_output_len):
        print(f"\n{'━' * 70}")
        print(f"  Step {i}: {'PREFILL' if i == 0 else 'DECODE'}")
        print(f"{'━' * 70}")

        if i == 0:
            print(f"  [reset] 清空 KV Cache (acl.rt.memset on Device)")
            session.reset()
            print(f"    session.run_times={session.run_times}")
            print(f"    model.real_kv_size={session.model.real_kv_size}")
            print(f"    model.input_pos={session.model.input_pos}")

        print(f"\n  [输入] current_input_ids shape={current_input_ids.shape}")
        if current_input_ids.shape[1] <= 20:
            print(f"         values={current_input_ids[0].tolist()}")
        else:
            print(f"         values (前10): {current_input_ids[0][:10].tolist()} ...")
            print(f"         values (后10): {current_input_ids[0][-10:].tolist()}")

        # 调用 session.run — 这是真正的推理
        print(f"\n  [session.run] 开始 ...")
        print(f"    调用前: real_kv_size={session.model.real_kv_size}, input_pos={session.model.input_pos}")
        t1 = time.time()
        logits = session.run(current_input_ids)
        elapsed = time.time() - t1
        print(f"    调用后: real_kv_size={session.model.real_kv_size}, input_pos={session.model.input_pos}")
        print(f"    耗时: {elapsed*1000:.1f}ms")
        print(f"    返回 logits shape: {logits.shape if logits is not None else None}")

        # 采样
        next_token = infer_engine.sample_logits(
            logits[0][-1:],
            "greedy",
            0.95,
            0
        )
        next_token = next_token.reshape(1, -1)

        token_id = next_token[0, 0]
        token_text = infer_engine.tokenizer.decode([token_id])
        print(f"\n  [采样] next_token_id={token_id}, decoded={token_text!r}")

        # 检查 EOS
        if token_id == infer_engine.tokenizer.eos_token_id:
            print(f"  [EOS] 遇到结束符，停止生成")
            break

        ids_list.append(int(token_id))
        current_input_ids = next_token

        # 打印当前已生成的完整文本
        generated_text = infer_engine.tokenizer.decode(ids_list)
        print(f"  [已生成文本] {generated_text!r}")

        # 打印 KV Cache 使用情况
        kv_used = session.model.real_kv_size
        kv_total = KV_CACHE_LENGTH
        print(f"  [KV Cache] {kv_used}/{kv_total} ({100*kv_used/kv_total:.1f}%)")

    # 5. 最终结果
    print(f"\n\n{'═' * 70}")
    print(f" 推理完成")
    print(f"{'═' * 70}")
    final_text = infer_engine.tokenizer.decode(ids_list)
    print(f"  输入 prompt: {TEST_PROMPT!r}")
    print(f"  输入 token 数: {input_length}")
    print(f"  生成 token 数: {len(ids_list)}")
    print(f"  生成内容: {final_text!r}")
    print(f"  KV Cache 最终使用: {session.model.real_kv_size}/{KV_CACHE_LENGTH}")
    print(f"  生成 token ids: {ids_list}")


if __name__ == "__main__":
    main()
