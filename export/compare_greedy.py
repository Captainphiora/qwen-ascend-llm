"""
Greedy 逐 token 对比 onnx(CPU) 与 om(NPU) 的完整生成结果。

对每一步:
  - onnx 与 om 各自用自己的 logits 做 argmax(greedy) 得到 next token
  - 打印两侧 token 是否一致
  - 关键: 为保证是"同一条轨迹上"的逐步对比, 两侧都用 onnx 选出的 token 作为下一步输入
    (teacher forcing), 这样即使某步 argmax 偶尔不同, 后续仍在同一上下文下比较。
"""
import os
import sys
import argparse
import numpy as np


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
# sys.path.append(project_dir)
sys.path.insert(0, project_dir)

from transformers import AutoTokenizer
from config import InferenceConfig
from utils.session import Session

parser = argparse.ArgumentParser()
parser.add_argument("--hf_model_dir", type=str, required=True)
parser.add_argument("--onnx_model_path", type=str, required=True)
parser.add_argument("--om_model_path", type=str, required=True)
parser.add_argument("--kv_cache_length", type=int, default=1024)
parser.add_argument("--max_prefill_length", type=int, default=1)
parser.add_argument("--cpu_thread", type=int, default=4)
parser.add_argument("--dtype", type=str, default="float16")
parser.add_argument("--prompt", type=str, default="背诵《出师表》")
parser.add_argument("--max_new_tokens", type=int, default=20)
parser.add_argument("--teacher_forcing", type=int, default=1,
                    help="1: 两侧共用 onnx 选出的 token 推进(逐步对比); 0: 各自独立生成")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)


def build_session(session_type):
    config = InferenceConfig(
        hf_model_dir=args.hf_model_dir,
        om_model_path=args.om_model_path,
        onnx_model_path=args.onnx_model_path,
        session_type=session_type,
        cpu_thread=args.cpu_thread,
        kv_cache_length=args.kv_cache_length,
        max_prefill_length=args.max_prefill_length,
        dtype=args.dtype,
    )
    return Session.fromConfig(config)


# 构造输入 token 序列
system_prompt = "You are a helpful assistant."
history = [{"role": "system", "content": system_prompt},
           {"role": "user", "content": args.prompt}]
text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer([text], return_tensors="np")["input_ids"].astype(np.int64)
print("prompt        :", args.prompt)
print("prompt tokens :", prompt_ids.shape[-1])

print("\n[init] loading onnx session (CPU) ...")
onnx_sess = build_session("onnx")
print("[init] loading om session (NPU) ...")
om_sess = build_session("acl")


def prefill(sess, ids):
    """把整个 prompt 喂进去, 返回最后一步的 logits"""
    logits = sess.run(ids)
    return logits


def greedy_next(logits):
    row = logits.reshape(-1, logits.shape[-1])[-1].astype(np.float32)
    return int(row.argmax())


# ---- prefill 阶段 ----
onnx_logits = prefill(onnx_sess, prompt_ids)
om_logits = prefill(om_sess, prompt_ids)
onnx_tok = greedy_next(onnx_logits)
om_tok = greedy_next(om_logits)
print("\n==== prefill 结束, 第 1 个待生成 token ====")
print("  ONNX: id={:<7} {!r}".format(onnx_tok, tokenizer.decode([onnx_tok])))
print("  OM  : id={:<7} {!r}".format(om_tok, tokenizer.decode([om_tok])))
print("  match:", "YES" if onnx_tok == om_tok else "NO")

# ---- decode 阶段 ----
eos_id = tokenizer.eos_token_id
match_cnt = 1 if onnx_tok == om_tok else 0
total = 1
onnx_text_tokens = [onnx_tok]
om_text_tokens = [om_tok]
print("\n==== 逐步 decode (teacher_forcing={}) ====".format(args.teacher_forcing))
print("step |  onnx_id onnx_tok       |   om_id om_tok         | match")
print(greedy_next.__doc__ and "-" * 70 or "-" * 70)

cur = onnx_tok if args.teacher_forcing else None
onnx_cur, om_cur = onnx_tok, om_tok
for step in range(1, args.max_new_tokens):
    if args.teacher_forcing:
        feed = np.array([[cur]], dtype=np.int64)
        onnx_logits = onnx_sess.run(feed)
        om_logits = om_sess.run(feed)
    else:
        onnx_logits = onnx_sess.run(np.array([[onnx_cur]], dtype=np.int64))
        om_logits = om_sess.run(np.array([[om_cur]], dtype=np.int64))
    onnx_tok = greedy_next(onnx_logits)
    om_tok = greedy_next(om_logits)
    total += 1
    same = onnx_tok == om_tok
    if same:
        match_cnt += 1
    print("{:>4} | {:>8} {:<14} | {:>8} {:<14} | {}".format(
        step, onnx_tok, repr(tokenizer.decode([onnx_tok]))[:14],
        om_tok, repr(tokenizer.decode([om_tok]))[:14],
        "YES" if same else "NO"))
    onnx_text_tokens.append(onnx_tok)
    om_text_tokens.append(om_tok)
    if args.teacher_forcing:
        cur = onnx_tok
        if onnx_tok == eos_id:
            break
    else:
        onnx_cur, om_cur = onnx_tok, om_tok
        if onnx_tok == eos_id and om_tok == eos_id:
            break

print("\n==== 汇总 ====")
print("对比步数        :", total)
print("greedy 一致步数 :", match_cnt)
print("一致率          : {:.2%}".format(match_cnt / total))
print("\nONNX 生成文本:\n", tokenizer.decode(onnx_text_tokens))
print("\nOM   生成文本:\n", tokenizer.decode(om_text_tokens))

om_sess.close()
onnx_sess.close()
