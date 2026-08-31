import sys
import math
import argparse
from concurrent.futures import ThreadPoolExecutor
from config import InferenceConfig
from utils.inference import Inference
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
SESSION_TYPE = "acl"
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR = f"/mnt/host-model/cxj/models/{MODEL_NAME}"
MAX_INPUT_LENGTH = 4095
MAX_OUTPUT_LENGTH = 4096
KV_CACHE_LENGTH = 4096
MAX_PREFILL_LENGTH = 1
OM_MODEL_PATH = f"./output/model_310_cann900/{MODEL_NAME}_{KV_CACHE_LENGTH}_{MAX_PREFILL_LENGTH}_sim.om"
CPU_THREAD = 8

# 原本未指定的其他参数，保留默认值
DTYPE = "float32"
TORCH_DTYPE = "float32"
DEVICE_STR = "npu"
MAX_BATCH = 1
ONNX_MODEL_PATH = os.path.join(project_dir, "output", "onnx", "qwen2_1.5b_chat.onnx")
# 采样与对话模板相关默认值（可被命令行覆盖；API 部署时再被单次请求覆盖）
# DeepSeek-R1 官方建议不使用 system prompt，故默认置空；温度推荐 0.6
TEMPERATURE = 0.6
SYSTEM_PROMPT = ""


def parser_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--hf_model_dir',
        type=str,
        help="model and tokenizer path, only support huggingface model",
        default=HF_MODEL_DIR
    )
    parser.add_argument(
        "--session_type",
        type=str,
        default=SESSION_TYPE,
        help="acl or onnx",
        choices=["acl", "onnx", "pytorch"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="support float16/float32, if use CPU, only support fp32",
        choices=["float16", "float32"],
        default=DTYPE,
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        help="support float16/float32, if use CPU, only support fp32",
        choices=["float16", "float32"],
        default=TORCH_DTYPE,
    )
    parser.add_argument(
        "--device_str",
        type=str,
        help="support cpu, cuda, npu, only activate when sesstion_type is pytorch",
        choices=["cpu", "cuda", "npu"],
        default=DEVICE_STR,
    )
    parser.add_argument(
        "--cpu_thread",
        type=int,
        help="num of cpu thread when run onnx sesstion",
        default=CPU_THREAD,
    )
    parser.add_argument(
        '--onnx_model_path',
        type=str,
        help="onnx_model_path",
        default=ONNX_MODEL_PATH
    )
    parser.add_argument(
        "--om_model_path",
        help="mindspore model path",
        type=str,
        default=OM_MODEL_PATH
    )
    parser.add_argument(
        "--max_batch",
        help="max batch",
        type=int,
        default=MAX_BATCH,
    )
    parser.add_argument(
        "--max_input_length",
        help="max input length",
        type=int,
        default=MAX_INPUT_LENGTH,
    )
    parser.add_argument(
        "--max_prefill_length",
        help="max prefill length in first inference. "
             "Attention max_prefill_length + max_output_length <= kv_cache_length. "
             "the number must by 2^xx, like 1, 2, 4, 8, 16, 32, 64, 128, 256... "
             "Note! The higher this number, the longer it will take to compile.",
        type=int,
        default=MAX_PREFILL_LENGTH,
    )
    parser.add_argument(
        "--max_output_length",
        help="max output length (contain input + new token)",
        type=int,
        default=MAX_OUTPUT_LENGTH,
    )
    parser.add_argument(
        "--temperature",
        help="sampling temperature; 设为 0 等价于 greedy。",
        type=float,
        default=TEMPERATURE,
    )
    parser.add_argument(
        "--sampling_method",
        help="sampling method: greedy, top_p, or top_k",
        type=str,
        default="top_p",
        choices=["greedy", "top_p", "top_k"],
      )
    parser.add_argument(
        "--sampling_value",
        help="sampling value: top_p range (0,1], top_k positive int",
        type=float,
        default=0.8
    ),
    parser.add_argument(
        "--system_prompt",
        help="system prompt; 以命令行为准：不传或传空字符串则不添加 system 消息，"
             "其余情况使用用户自定义的提示词。",
        type=str,
        default=SYSTEM_PROMPT,
    ),
    parser.add_argument(
        "--sampling_device",
        help="sampling device: cpu uses numpy sampling, npu uses torch_npu on-device sampling (zero-copy)",
        type=str,
        default="cpu",
        choices=["cpu", "npu"],
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
    )
    return parser.parse_args()


def inference_cli(config):
    infer_engine = Inference(config)
    print("\n欢迎使用Qwen聊天机器人，输入exit或者quit退出，输入clear清空历史记录")
    history = []
    while True:
        input_text = input("Input: ")
        if input_text in ["exit", "quit", "exit()", "quit()"]:
            break
        if input_text == 'clear':
            history = []
            infer_engine.session.reset()
            print("Output: 已清理历史对话信息。")
            continue
        print("Output: ", end='')
        response = ""
        is_first = True
        first_token_lantency, decode_speed, total_speed = 0, 0, 0.0
        for (
                new_text,
                first_token_lantency,
                decode_speed,
                total_speed
            ) in infer_engine.stream_predict(input_text, history=history, do_speed_test=True):
            if is_first:
                if len(new_text.strip()) == 0:
                    continue
                is_first = False
            print(new_text, end='', flush=True)
            response += new_text
        print("")
        print(
            "[INFO] first_token_lantency: {:.4f}s,".format(first_token_lantency),
            " decode_speed: {:.2f} token/s, ".format(decode_speed),
            " total_speed(prefill+decode): {:.2f} token/s".format(total_speed),
        )
        history.append([input_text, response])


def main_cli():
    args = parser_args()
    max_prefill_log2 = int(math.log2(args.max_prefill_length))
    max_prefill_length = 2 ** max_prefill_log2
    config = InferenceConfig(
        hf_model_dir=args.hf_model_dir,
        om_model_path=args.om_model_path,
        onnx_model_path=args.onnx_model_path,
        cpu_thread=args.cpu_thread,
        session_type=args.session_type,
        max_batch=args.max_batch,
        max_output_length=args.max_output_length,
        max_input_length=args.max_input_length,
        kv_cache_length=args.max_output_length,
        max_prefill_length=max_prefill_length,
        dtype=args.dtype,
        torch_dtype=args.torch_dtype,
        temperature=args.temperature,
        sampling_method=args.sampling_method,
        sampling_value=args.sampling_value,
        system_prompt=args.system_prompt,
        device_str=args.device_str,
        device_id=args.device_id,
        sampling_device=args.sampling_device,
    )
    print("==================== 实际生效的推理配置(config) ====================")
    print("session_type      : {}".format(config.session_type))
    print("hf_model_dir      : {}".format(config.hf_model_dir))
    print("tokenizer_dir     : {}".format(config.tokenizer_dir))
    print("om_model_path     : {}".format(config.om_model_path))
    print("onnx_model_path   : {}".format(config.onnx_model_path))
    print("device_str        : {}".format(config.device_str))
    print("device_id         : {}".format(config.device_id))
    print("cpu_thread        : {}".format(config.cpu_thread))
    print("max_batch         : {}".format(config.max_batch))
    print("max_input_length  : {}".format(config.max_input_length))
    print("max_output_length : {}".format(config.max_output_length))
    print("max_prefill_length: {}".format(config.max_prefill_length))
    print("kv_cache_length   : {}".format(config.kv_cache_length))
    print("kvcache_method    : {}".format(config.kvcache_method))
    print("cache_format      : {}".format(config.cache_format))
    print("dtype             : {}".format(config.dtype))
    print("torch_dtype       : {}".format(config.torch_dtype))
    print("-------------------- 采样相关(是否 greedy) --------------------")
    print("sampling_method   : {}".format(config.sampling_method))
    print("sampling_value : {}".format(config.sampling_value))
    print("sampling_device   : {}".format(config.sampling_device))
    print("temperature       : {}".format(config.temperature))
    is_greedy = (config.temperature == 0) or (config.sampling_method == "greedy")
    print("=> 实际是否 greedy : {}".format(is_greedy))
    print("-------------------- 提示词 --------------------")
    print("system_prompt     : {!r}  (空字符串表示不添加 system 消息)".format(config.system_prompt))
    print("================================================================")
    inference_cli(config)


if __name__ == '__main__':
    main_cli()
