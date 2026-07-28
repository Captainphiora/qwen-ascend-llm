# DeepSeek-R1-Distill 模型使用说明

## 模型特性

DeepSeek-R1-Distill 系列模型使用了特殊的"思维链"（Chain of Thought）格式，会在输出最终答案前先进行推理。

### 输出格式

```
<think>
[模型的内部推理过程]
第一步：理解问题...
第二步：分析选项...
第三步：得出结论...
</think>
[最终答案]
```

## 代码修改

为了提供更好的用户体验，我们在 `utils/inference.py` 中实现了自动过滤思维链的功能：

1. **自动隐藏思维过程**：用户界面只显示 `</think>` 之后的最终答案
2. **兼容普通模型**：对 Qwen2.5 等普通模型无影响
3. **流式生成优化**：思考阶段不输出任何内容，只在完成思考后才开始流式输出

### 修改的方法

- `stream_predict()`: 流式生成，支持逐 token 输出
- `predict()`: 非流式生成，一次性返回完整结果

## 使用示例

### 正常对话

```bash
python3 cli_chat.py \
  --session_type=acl \
  --hf_model_dir=/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --om_model_path=/home/chenxinji/qwen-ascend-llm/output/model/DeepSeek-R1-Distill-Qwen-1.5B_1024_1.om \
  --max_input_length=1024 \
  --max_output_length=1024 \
  --max_prefill_length=1
```

### 用户体验

**输入**：
```
Input: 1+1等于几？
```

**输出**（用户看到的）：
```
Output: 1+1等于2。
[INFO] first_token_lantency: 0.0523s, decode_speed: 45.32 token/s, total_speed(prefill+decode): 38.76 token/s, input_tokens: 23, output_tokens: 15
```

**实际生成的完整内容**（被过滤掉的部分）：
```
<think>
这是一个简单的加法问题。
1加1的结果是2。
</think>
1+1等于2。
```

## 特殊 Token

DeepSeek-R1-Distill 模型的特殊 token：

| Token | ID | 说明 |
|-------|-------|------|
| `<｜begin▁of▁sentence｜>` | 151646 | 句子开始标记（BOS） |
| `<｜end▁of▁sentence｜>` | 151643 | 句子结束标记（EOS） |
| `<｜User｜>` | 151644 | 用户角色标记 |
| `<｜Assistant｜>` | 151645 | 助手角色标记 |
| `<think>` | 151648 | 思维链开始标记 |
| `</think>` | 151649 | 思维链结束标记 |

## 与 Qwen2.5 的区别

| 特性 | Qwen2.5-1.5B-Instruct | DeepSeek-R1-Distill-Qwen-1.5B |
|------|----------------------|-------------------------------|
| EOS Token | `<|im_end|>` (151645) | `<｜end▁of▁sentence｜>` (151643) |
| Chat Template | `<|im_start|>...<|im_end|>` | `<｜User｜>...<｜Assistant｜>` |
| 思维链 | 无 | 有 `<think>...</think>` |
| 输出风格 | 直接回答 | 先思考后回答 |

## 高级配置

### 显示思维过程（用于调试）

如果你想看到完整的思维过程，可以临时注释掉过滤逻辑：

```python
# 在 utils/inference.py 的 stream_predict 方法中
# 注释掉第 220-233 行的过滤代码
```

### Token 统计

修改后的代码会统计：
- **input_tokens**: 包含 system prompt、历史对话和当前输入的总 token 数
- **output_tokens**: 模型生成的 token 数（包括思维链部分）

注意：即使思维链被隐藏，output_tokens 仍然包含 `<think>...</think>` 中的 token。

## 故障排查

### 问题 1：输出仍然包含 `<think>` 标签

**可能原因**：
- 代码修改未生效，检查是否正确编辑了 `utils/inference.py`
- Python 模块缓存问题，尝试删除 `__pycache__` 目录

**解决方法**：
```bash
cd /home/chenxinji/qwen-ascend-llm
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
python3 cli_chat.py ...
```

### 问题 2：生成速度明显变慢

**可能原因**：
- R1 模型会先生成思维链再生成答案，总 token 数更多
- 思维链长度通常是最终答案的 2-5 倍

**正常现象**：
- first_token_latency 可能较高（等待思维链生成完成）
- decode_speed 正常
- 总生成 token 数（output_tokens）会明显多于可见文本

### 问题 3：某些问题没有输出

**可能原因**：
- 模型生成了 `<think>` 但没有生成 `</think>` 就遇到 EOS
- max_output_length 太小，思维链被截断

**解决方法**：
```bash
# 增加 max_output_length
--max_output_length=2048
```

## 性能建议

1. **KV-cache 设置**：R1 模型的思维链会占用更多 KV-cache，建议 `kv_cache_length >= 1024`
2. **Max output length**：建议设置为思维链 + 答案的总长度，通常 1024-2048
3. **StreamingLLM**：如果需要长对话，启用 StreamingLLM 策略

## 参考资源

- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1-Distill 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
