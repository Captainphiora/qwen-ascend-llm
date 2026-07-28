# CLAUDE.md

This file provides guidance to AI assistant when working with code in this repository.

## Project Overview

This project enables Qwen1.5/Qwen2 LLM inference on Ascend NPU hardware (tested on Ascend 310B1). It converts HuggingFace models to ONNX format, then to Ascend's .om format for optimized NPU execution. The inference engine supports three backends: PyTorch (CPU/NPU), ONNX (CPU), and ACL (Ascend NPU).

**Key Architecture Points:**
- **Three-stage pipeline**: HuggingFace → ONNX → .om (Ascend model)
- **Three inference backends**: `pytorch`, `onnx`, `acl` (session types)
- **KV-cache management**: Supports fixsize, basic, streamllm, H2O methods (see `utils/kvcache.py`)
- **Dynamic shape inference**: Reduces first-token latency by decomposing prefill into power-of-2 chunks
- **OpenAI-compatible API**: FastAPI server at `api.py` with streaming support

## Common Commands

### Model Export and Compilation

Export HuggingFace model to ONNX:
```bash
python3 export/export_onnx.py \
  --device_str=npu \
  --dtype=float16 \
  --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
  --onnx_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
  --kv_cache_length=2048
```

Modify ONNX structure (fix Trilu operator for ATC):
```bash
python3 export/change_node.py \
  --input_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
  --output_model_path="./output/onnx2/qwen2_1.5b_chat.onnx"
```

Convert ONNX to .om (Ascend model):
```bash
python3 export/onnx2om.py \
  --onnx_model_path="./output/onnx2/qwen2_1.5b_chat.onnx" \
  --om_model_path="./output/model/qwen2_1.5b_chat.om" \
  --soc_version=Ascend310B1 \
  --max_prefill_length=4
```

### Testing and Validation

Verify PyTorch model structure (CPU):
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

Test ONNX inference (CPU):
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

Test ACL inference (Ascend NPU):
```bash
python3 ./cli_chat.py \
  --session_type=acl \
  --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
  --om_model_path="./output/model/qwen2_1.5b_chat.om" \
  --max_input_length=1024 \
  --max_output_length=2048 \
  --max_prefill_length=4
```

### API Server

Start OpenAI-compatible API server:
```bash
python3 api.py \
  --session_type=acl \
  --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
  --om_model_path="./output/model/qwen2_1.5b_chat.om" \
  --max_input_length=1024 \
  --max_output_length=2048 \
  --max_prefill_length=4
```

Test API clients (in `client/` directory):
- `openai_stream_client.py` - Streaming inference
- `openai_normal_client.py` - Non-streaming inference
- `openai_function_call.py` - Function calling support

### Debugging

Compare ONNX vs .om layer-by-layer outputs (requires msit tool):
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

## Key Implementation Details

### Session Architecture (`utils/session.py`)
Three session types inherit from base `Session` class:
- **OnnxSession**: CPU inference via ONNX Runtime
- **PyTorchSession**: Direct PyTorch inference (CPU/CUDA/NPU)
- **AclSession**: Ascend NPU inference via ACL runtime
  - Implements dynamic shape by decomposing input into power-of-2 chunks
  - Uses `max_prefill_length` to determine dynamic shape boundaries

### Inference Engine (`utils/inference.py`)
Core class `Inference` wraps sessions and provides:
- Tokenization with chat template formatting
- Sampling methods: greedy, top_p, top_k
- Streaming generation with progress tracking
- KV-cache management through session delegation

### ACL Engine (`utils/engine.py`)
Low-level Ascend NPU interface:
- `ACLModel`: Loads .om files, manages device memory, executes inference
- `init_resource()`: Initializes ACL runtime and device context
- Implements zero-copy memory mapping for model weights
- Supports async execution with callbacks

### KV-Cache (`utils/kvcache.py`)
Multiple strategies implemented:
- `fixsize`: Fixed-size circular buffer (default)
- `basic`: Basic cache without eviction
- `streamllm`: StreamingLLM-style attention sinking
- `H2O`: Heavy-Hitter Oracle eviction

### Model Export (`export/export_onnx.py`)
- Copies modified `modeling_qwen2.py` to model directory
- Exports with dynamic axes for batch_size and seq_length
- Critical: sets `kv_cache_length` at export time (immutable later)

## Important Constraints

1. **KV-cache length**: Set during ONNX export, cannot be changed later. Must satisfy:
   - `max_input_length < kv_cache_length`
   - `max_output_length == kv_cache_length` (for ONNX/ACL sessions)
   - `max_prefill_length + max_output_length <= kv_cache_length`

2. **max_prefill_length**: Must be power of 2 (1, 2, 4, 8, 16, ...). Higher values increase compilation time but may reduce first-token latency.

3. **dtype consistency**:
   - NPU export: use `float16`
   - CPU export: use `float32`
   - ACL inference: requires NPU-exported models (float16)

4. **SOC version**: Must match target hardware (e.g., `Ascend310B1` for Orange Pi AI Pro)

## File Structure Notes

- `config.py`: Central configuration class `InferenceConfig`
- `cli_chat.py`: Interactive CLI chat interface
- `api.py`: FastAPI server with OpenAI-compatible endpoints
- `export/`: Model conversion pipeline scripts
- `utils/`: Core inference engine, session management, KV-cache
- `client/`: Example API clients
- `ops_info.json`: Custom operator definitions for ATC compilation

## Docker Deployment

Build deployment image (requires .om file):
```bash
docker build . -t qwen_ascend_llm
```

Build development image (for custom model compilation):
```bash
docker build -f Dockerfile_dev . -t qwen_ascend_llm_dev
```

Run container:
```bash
./run_container.sh
```

## Troubleshooting

If .om inference produces incorrect output while ONNX works:
1. Use `export/compare.py` with msit tool to identify problematic layers
2. Consider using static shape (max_prefill_length=1) for easier debugging
3. Test with smaller models first (Qwen2-0.5B-Instruct recommended)
4. Check that 'mixed_float16' operators are not forcing incorrect layers to FP16
