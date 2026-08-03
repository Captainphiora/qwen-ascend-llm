# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project enables Qwen1.5/Qwen2 language models to run on Ascend NPU devices (tested on Ascend 310B1). It supports model conversion from PyTorch → ONNX → OM (Ascend optimized model) and provides inference capabilities through ONNX (CPU), PyTorch (CPU/NPU), and ACL (Ascend C Language, NPU-optimized).

The codebase is adapted from the [ascend-llm](https://gitee.com/yinghuo302/ascend-llm) project.

## Core Architecture

### Three Inference Session Types
The system supports three runtime backends controlled by `session_type` parameter:
- **`acl`**: Ascend NPU inference using compiled OM models (production path)
- **`onnx`**: ONNX Runtime on CPU (validation/debugging)
- **`pytorch`**: PyTorch on CPU/NPU (model structure verification)

All three share the same inference pipeline through `utils/inference.py` and KV-cache management in `utils/kvcache.py`.

### Model Pipeline: PyTorch → ONNX → OM
1. **Export ONNX** (`export/export_onnx.py`): Converts HuggingFace Qwen models to ONNX with fixed KV-cache length
2. **Modify ONNX** (`export/change_node.py`): Fixes Trilu operators incompatible with Ascend ATC compiler
3. **Compile OM** (`export/onnx2om.py`): Uses Ascend ATC to compile ONNX → OM with dynamic batch/sequence support

### KV-Cache System (`utils/kvcache.py`)
Manages four cache strategies via `kvcache_method`:
- `fixsize`: Fixed-size cache (default)
- `basic`: Basic cache management
- `streamllm`: StreamLLM approach
- `H2O`: Heavy-Hitter Oracle

Cache shape: `(batch, kv_cache_length, num_layers * 2 * num_kv_heads, head_dim)`

### Configuration (`config.py`)
Central `InferenceConfig` class holds:
- Model paths (HuggingFace dir, ONNX path, OM path)
- Runtime params (max_input_length, max_output_length, kv_cache_length)
- Session type and device settings
- Sampling parameters (method, temperature, top_p/top_k)

## Common Development Commands

### Setup and Installation
```bash
# Install dependencies (requires Python >= 3.10)
pip install -r requirements.txt

# Or use uv (if available)
uv pip install -r requirements.txt
```

### Model Export Pipeline

#### Step 1: Export to ONNX
```bash
python3 export/export_onnx.py \
  --device_str=npu \
  --dtype=float16 \
  --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
  --onnx_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
  --kv_cache_length=2048
```

#### Step 2: Fix ONNX Node Compatibility
```bash
python3 export/change_node.py \
  --input_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
  --output_model_path="./output/onnx2/qwen2_1.5b_chat.onnx"
```

#### Step 3: Compile to OM (Ascend optimized model)
```bash
python3 export/onnx2om.py \
  --onnx_model_path="./output/onnx2/qwen2_1.5b_chat.onnx" \
  --output_model_path="./output/model/qwen2_1.5b_chat.om" \
  --soc_version=Ascend310B1 \
  --max_prefill_length=8
```

### Testing and Validation

#### Verify PyTorch Model Structure
```bash
python3 ./cli_chat.py \
    --session_type="pytorch" \
    --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
    --device_str="cpu" \
    --dtype="float32" \
    --torch_dtype="float32" \
    --max_input_length=1024 \
    --max_output_length=2048
```

#### Verify ONNX Model
```bash
python3 ./cli_chat.py \
  --session_type=onnx \
  --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
  --onnx_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
  --dtype="float16" \
  --cpu_thread=4 \
  --max_input_length=1024 \
  --max_output_length=2048
```

#### Test OM on Ascend NPU
```bash
python3 ./cli_chat.py \
  --session_type="acl" \
  --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
  --om_model_path="./output/model/qwen2_1.5b_chat.om" \
  --device_id=0 \
  --dtype="float16" \
  --max_input_length=1024 \
  --max_output_length=2048 \
  --max_prefill_length=8
```

### Model Comparison and Debugging
Compare ONNX vs OM layer-by-layer outputs using MSIT tool:
```bash
python3 export/compare.py \
  --hf_model_dir="./download/Qwen2-0.5B-Instruct" \
  --onnx_model_path="./output/onnx2/qwen2_0.5b_chat.onnx" \
  --om_model_path="./output/model/qwen2_0.5b_chat.om" \
  --kv_cache_length=2048 \
  --cpu_thread=1 \
  --dtype="float16" \
  --max_prefill_length=1
```

### API Server

#### Start OpenAI-compatible API server
```bash
python3 api.py \
  --session_type="acl" \
  --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
  --om_model_path="./output/model/qwen2_1.5b_chat.om" \
  --dtype="float16" \
  --max_input_length=1024 \
  --max_output_length=2048 \
  --max_prefill_length=8 \
  --device_id=0
```

Server will run on `http://0.0.0.0:8000` with OpenAI-compatible endpoints.

#### Test API clients
```bash
cd client
# Streaming response
python3 openai_stream_client.py

# Non-streaming response
python3 openai_normal_client.py

# Function calling
python3 openai_function_call.py
```

### Docker Deployment

#### Build deployment Docker image
```bash
docker build . -t qwen_ascend_llm
```

#### Build development Docker image
```bash
docker build -f Dockerfile_dev . -t qwen_ascend_llm_dev
```

#### Run container
```bash
./run_container.sh
```

## Critical Parameter Constraints

### KV-Cache Length Consistency
The `kv_cache_length` parameter **must be consistent** across the entire pipeline:
- Set during ONNX export (`export_onnx.py --kv_cache_length`)
- Used in runtime inference (`cli_chat.py --max_output_length`)
- `max_output_length` at runtime must equal the `kv_cache_length` used during export

### Input/Output Length Rules
- `max_input_length` < `kv_cache_length` (typically kv_cache_length / 2)
- `max_output_length` = `kv_cache_length` 
- Actual max generation tokens = `max_output_length` - min(`max_input_length`, actual_input_tokens)

### Dynamic Batch/Sequence Support
The `max_prefill_length` parameter controls dynamic shape compilation:
- Must be power of 2 (e.g., 1, 2, 4, 8, 16)
- Larger values reduce first-token latency but increase memory usage
- OM compilation creates multiple shape variants: [1, 2, 4, ..., max_prefill_length]

### Data Type Constraints
- NPU/CUDA ONNX export: `dtype=float16`
- CPU ONNX export: `dtype=float32`
- ACL session: always `float16`

## Code Organization

### Entry Points
- `api.py`: FastAPI server with OpenAI-compatible endpoints
- `cli_chat.py`: Interactive CLI chat interface
- `main.py`: Minimal entry point

### Export Pipeline (`export/`)
- `modeling_qwen2.py`: Modified Qwen2 architecture (from transformers v4.37.0)
- `export_onnx.py`: PyTorch → ONNX conversion
- `change_node.py`: ONNX graph surgery (fix Trilu nodes)
- `onnx2om.py`: ONNX → OM compilation via ATC
- `compare.py`: Layer-wise output comparison (ONNX vs OM)

### Runtime (`utils/`)
- `session.py`: Session abstractions (`OnnxSession`, `PyTorchSession`, `AclSession`)
- `inference.py`: Unified inference pipeline and sampling logic
- `kvcache.py`: KV-cache management strategies
- `engine.py`: Low-level ACL runtime wrapper

### Configuration
- `config.py`: `InferenceConfig` dataclass with all runtime parameters
- `pyproject.toml`: Python dependencies (managed by pip/uv)

## Key Implementation Details

### Custom Qwen2 Model (`export/modeling_qwen2.py`)
Modified from HuggingFace transformers v4.37.0 to:
- Support static KV-cache shapes required by ONNX export
- Handle position_ids and attention_mask for incremental decoding
- Maintain compatibility with original HuggingFace checkpoints

### ACL Session (`utils/session.py:AclSession`)
Wraps Ascend CANN ACL runtime:
- Dynamic batch/sequence dispatch based on input size
- Manages NPU memory and model execution
- Interfaces with `utils/engine.py` for low-level ACL calls

### Inference Pipeline (`utils/inference.py`)
Unified across all session types:
- Tokenization with HuggingFace tokenizers
- Sampling: greedy / top_p / top_k
- Streaming output support
- Special token handling (stop tokens, EOS)

## Important Notes

### Model Compatibility
- Tested: Qwen1.5-0.5B-Chat, Qwen2-1.5B-Instruct
- Should support: All Qwen1.5/Qwen2 chat/instruct variants
- Requires: HuggingFace format checkpoints (not GGUF/other formats)

### CANN Environment
- Requires CANN 8.0 RC2 or higher
- Must set NPU device via `--device_id` (default 0)
- ATC compilation requires matching `--soc_version` (e.g., Ascend310B1)

### Debugging FP16 Issues
If OM inference produces garbage output while ONNX works:
1. Use `export/compare.py` to identify divergent layers
2. Check if specific layers need `mixed_float16` precision override
3. Use MSIT tool for detailed layer-wise comparison
4. Consider using static shape (max_prefill_length=1) for easier debugging

### Test Prompts
When verifying model correctness, use challenging prompts like:
- "背诵《出师表》" (Recite "Chu Shi Biao" - tests memorization)
- Multi-turn conversations
- Code generation tasks
