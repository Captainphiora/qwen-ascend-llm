import os
import sys
from datetime import datetime
import ctypes
import subprocess
import argparse
import math
from transformers.models.qwen2 import Qwen2Config


class TeeLogger:
    """Duplicate stdout to both console and a log file."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
onnx_model_dir = os.path.join(output_dir, "onnx2")
if not os.path.exists(onnx_model_dir):
    os.mkdir(onnx_model_dir)
model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--soc_version',
    type=str,
    default="auto",
    help="NPU full name, like Ascend310B1、Ascend310B4、Ascend310P1、Ascend910A、Ascend910B..., default is `auto`, will auto detect soc version.",
)
parser.add_argument(
    '--hf_model_dir',
    type=str,
    help="model and tokenizer path, only support huggingface model",
    default=os.path.join(project_dir, "download", "Qwen2-1.5B-Instruct")
)
parser.add_argument(
    "--onnx_model_path",
    help="output onnx path",
    type=str,
    default=os.path.join(onnx_model_dir, "qwen2_1.5b_chat.onnx")
)
parser.add_argument(
    "--om_model_path",
    help=".om model path",
    type=str,
    default= os.path.join(model_dir, "qwen2_1.5b_chat")
)
parser.add_argument(
    "--max_batch",
    help="max batch",
    type=int,
    default=1,
)
parser.add_argument(
    "--cpu_thread" ,
    type=int,
    help="num of cpu thread when convert onnx to om",
    default=1,
)
parser.add_argument(
    "--max_prefill_length",
    help="max prefill length in first inference. "
        "Attention max_prefill_length + max_output_length <= kv_cache_length. "
        "the number must by 2^xx, like 1, 2, 4, 8, 16, 32, 64, 128, 256... "
        "Note! The higher this number, the longer it will take to compile.",
    type=int,
    default=8
)
parser.add_argument(
    "--kv_cache_length",
    help="kv-cache length",
    type=int,
    default=2048,
)
parser.add_argument(
    "--enable_compress_weight",
    help="enable weight compression for bandwidth optimization (Ascend 910 only)",
    type=str,
    choices=["true", "false"],
    default="false",
)
parser.add_argument(
    "--compression_optimize_conf",
    help="path to compression_optimize.cfg for calibration-based PTQ",
    type=str,
    default="",
)
parser.add_argument(
    "--disable_constant_folding",
    help="disable ATC constant folding (needed for quantized models with INT8 weights)",
    type=str,
    choices=["true", "false"],
    default="false",
)


args = parser.parse_args()

log_dir = os.path.join(project_dir, "onnx_log")
om_basename = os.path.basename(args.om_model_path)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"onnx2om_{om_basename}_{timestamp}.log"
log_path = os.path.join(log_dir, log_filename)

tee = TeeLogger(log_path)
sys.stdout = tee

try:

    def get_soc_version():
        max_len = 512
        rtsdll = ctypes.CDLL(f"libruntime.so")
        c_char_t = ctypes.create_string_buffer(b"\xff" * max_len, max_len)
        rtsdll.rtGetSocVersion.restype = ctypes.c_uint64
        rt_error = rtsdll.rtGetSocVersion(c_char_t, ctypes.c_uint32(max_len))
        if rt_error:
            print("rt_error:", rt_error)
            return ""
        soc_full_name = c_char_t.value.decode("utf-8")
        find_str = "Short_SoC_version="
        ascend_home_dir = os.environ.get("ASCEND_HOME_PATH")
        assert ascend_home_dir is not None, \
            print("ASCEND_HOME_PATH is None, you need run `source /usr/local/Ascend/ascend-toolkit/set_env.sh`")
        platform_config_candidates = [
            f"{ascend_home_dir}/compiler/data/platform_config/{soc_full_name}.ini",
            f"{ascend_home_dir}/aarch64-linux/data/platform_config/{soc_full_name}.ini",
            f"{ascend_home_dir}/x86_64-linux/data/platform_config/{soc_full_name}.ini",
        ]
        platform_config_path = None
        for candidate in platform_config_candidates:
            if os.path.isfile(candidate):
                platform_config_path = candidate
                break
        if platform_config_path is None:
            raise FileNotFoundError(
                f"Cannot find platform config for {soc_full_name}, searched: {platform_config_candidates}")
        with open(platform_config_path, "r") as f:
            for line in f:
                if find_str in line:
                    start_index = line.find(find_str)
                    soc_short_name = line[start_index + len(find_str):].strip()
                    return {
                       "soc_full_name": soc_full_name,
                       "soc_short_name": soc_short_name
                    }
        raise Exception("can't get you soc version")

    max_batch = args.max_batch
    model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
    num_hidden_layers = model_config.num_hidden_layers
    num_key_value_heads = model_config.num_key_value_heads
    hidden_size = model_config.hidden_size
    num_attention_heads = model_config.num_attention_heads
    per_head_dim = hidden_size // num_attention_heads
    kv_cache_length = args.kv_cache_length
    max_prefill_log2 = int(math.log2(args.max_prefill_length))
    max_prefill_length = 2 ** max_prefill_log2
    prefill_length_range = list(range(0, max_prefill_log2 + 1))
    prefill_length_range = [2 ** idx for idx in prefill_length_range]
    assert (max_prefill_length < kv_cache_length), \
        print("max_input_length max be smaller than kv_cache_length, because max_input_length + max_output_length <= kv_cache")
    input_ids_length_range = prefill_length_range
    position_length_range = prefill_length_range
    input_ids_shape = [
        f"1~{max_batch}" if max_batch > 1 else "1",
        "-1" if max_prefill_length > 1 else "1",
    ]
    attention_mask_shape = [
        f"1~{max_batch}" if max_batch > 1 else "1",
        "-1" if max_prefill_length > 1 else str(1 + kv_cache_length)
    ]
    position_ids_shape = [
        f"1~{max_batch}" if max_batch > 1 else "1",
        "-1" if max_prefill_length > 1 else "1"
    ]
    past_key_values_shape = [
        f"1~{max_batch}" if max_batch > 1 else "1",
        "-1" if max_prefill_length > 1 else kv_cache_length,
        num_hidden_layers * 2 * num_key_value_heads,
        per_head_dim
    ]
    dynamic_dims = []
    for dynamic_kv_cache_length in [
        kv_cache_length // 2,
        kv_cache_length
    ]:
        for dynamic_dim in zip(input_ids_length_range, position_length_range):
            new_dynamic_dim = [
                str(dynamic_dim[0]),
                str(dynamic_dim[0] + dynamic_kv_cache_length),
                str(dynamic_dim[1]),
                str(dynamic_kv_cache_length),
            ]
            dynamic_dims.append(",".join(new_dynamic_dim))

    past_key_values_shape = [str(x) for x in past_key_values_shape]
    if args.soc_version == "auto":
        print("[INFO] soc_version is `auto`, will auto detect soc version")
        soc_dict = get_soc_version()
        print("[INFO] {}".format(soc_dict))
        soc_version = soc_dict["soc_full_name"]
    else:
        soc_version = args.soc_version
    command_lines = [
        "export MS_DEV_FORCE_ACL=1 && ",
        "export MS_ENABLE_GE=1 && ",
        "export TE_PARALLEL_COMPILER={} &&".format(args.cpu_thread),
        "export MAX_COMPILE_CORE_NUMBER={} &&".format(args.cpu_thread),
        "atc",
        "--framework=5",
        '--model="{}"'.format(args.onnx_model_path),
        '--output="{}"'.format(args.om_model_path),
        "--soc_version={}".format(soc_version),
        "--precision_mode_v2=mixed_float16",
        "--modify_mixlist={}".format(os.path.join(project_dir, "ops_info.json")),
        "--input_format=ND",
        '--input_shape="input_ids:{};attention_mask:{};position_ids:{};past_key_values:{}"'.format(
            ",".join(input_ids_shape),
            ",".join(attention_mask_shape),
            ",".join(position_ids_shape),
            ",".join(past_key_values_shape)
        ),
    ]
    if args.enable_compress_weight == "true":
        command_lines.append("--enable_compress_weight=true")
    if args.compression_optimize_conf:
        command_lines.append(
            "--compression_optimize_conf={}".format(args.compression_optimize_conf)
        )
    if args.disable_constant_folding == "true":
        command_lines.append("--oo_constant_folding=false")
    if max_prefill_length > 1:
        command_lines.append(
            "--dynamic_dims \"{}\"".format(";".join(dynamic_dims))
        )
    print("============ run command ==============")
    print(" ".join(command_lines))
    print("=======================================")
    process = subprocess.Popen(
        " ".join(command_lines),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )
    for line in process.stdout:
        sys.stdout.write(line)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, " ".join(command_lines))
finally:
    print(f"\n[LOG] Log saved to: {log_path}")
    sys.stdout = tee.terminal
    tee.close()
