"""
Step 1: W8A8 PTQ Quantization using msmodelslim
Usage:
    python step1_quantize.py \
        --model_path /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
        --save_directory ./output/quant_w8a8 \
        --device_type npu \
        --device_id 0 \
        --calib_file ./calib_data/boolq.jsonl \
        --num_calibration_samples 50
"""

import os
import sys
import argparse
import json
import shutil
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_calib_texts(calib_file, num_samples=50):
    """Load raw text from JSONL calibration file."""
    if not os.path.exists(calib_file):
        print(f"[ERROR] Calibration file not found: {calib_file}")
        sys.exit(1)

    texts = []
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= num_samples:
                break
            data = json.loads(line.strip())
            if "inputs_pretokenized" in data:
                texts.append(data["inputs_pretokenized"])
            elif "passage" in data:
                texts.append(data["passage"])
            elif "text" in data:
                texts.append(data["text"])
            else:
                texts.append(str(list(data.values())[-1]))

    print(f"[INFO] Loaded {len(texts)} calibration samples from {calib_file}")
    return texts


def tokenize_calib_data(texts, tokenizer, device_type="npu"):
    """Tokenize calibration texts into [[input_ids, attention_mask], ...] format."""
    tokenized_data = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device_type)
        attention_mask = inputs["attention_mask"].to(device_type)
        tokenized_data.append([input_ids, attention_mask])
    return tokenized_data


def main():
    parser = argparse.ArgumentParser(description="W8A8 PTQ Quantization via msmodelslim")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--save_directory", type=str, required=True,
                        help="Output directory for quantized model")
    parser.add_argument("--device_type", type=str, default="npu",
                        choices=["npu", "cpu"],
                        help="Device for calibration: npu or cpu")
    parser.add_argument("--device_id", type=int, default=0,
                        help="NPU device ID")
    parser.add_argument("--calib_file", type=str, required=True,
                        help="Path to calibration data JSONL file")
    parser.add_argument("--num_calibration_samples", type=int, default=50,
                        help="Number of calibration samples to use")
    parser.add_argument("--w_bit", type=int, default=8,
                        help="Weight quantization bits")
    parser.add_argument("--a_bit", type=int, default=8,
                        help="Activation quantization bits")
    parser.add_argument("--anti_method", type=str, default="m4",
                        help="Anti-outlier method: m1, m2, m3, m4, or empty to disable")
    parser.add_argument("--disable_names", type=str, nargs="+",
                        default=["lm_head"],
                        help="Layer names to skip quantization")
    parser.add_argument("--act_method", type=int, default=1, choices=[1, 2, 3],
                        help="Activation quant method: 1=MinMax, 2=Histogram, 3=Auto")
    parser.add_argument("--do_smooth", action="store_true", default=False,
                        help="Enable SmoothQuant")
    parser.add_argument("--disable_level", type=str, default="L0",
                        help="Disable level for calibrator")
    args = parser.parse_args()

    if args.device_type == "npu":
        import torch_npu
        torch_npu.npu.set_device(args.device_id)
        torch.npu.set_compile_mode(jit_compile=False)
        print(f"[INFO] Using NPU device {args.device_id}")

    from transformers import AutoTokenizer
    from transformers.models.qwen2 import Qwen2ForCausalLM
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
    from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig

    print(f"[INFO] Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"[INFO] Loading model from: {args.model_path}")
    if args.device_type == "npu":
        model = Qwen2ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = Qwen2ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    print(f"[INFO] Model loaded on {args.device_type}")

    print(f"[INFO] Preparing calibration data...")
    calib_texts = load_calib_texts(args.calib_file, num_samples=args.num_calibration_samples)
    calib_data = tokenize_calib_data(calib_texts, tokenizer, device_type=args.device_type)
    print(f"[INFO] Tokenized {len(calib_data)} calibration samples")

    if args.anti_method:
        print(f"[INFO] Running AntiOutlier with method: {args.anti_method}")
        anti_config = AntiOutlierConfig(
            anti_method=args.anti_method,
            dev_type=args.device_type,
            dev_id=args.device_id if args.device_type == "npu" else 0,
        )
        anti_outlier = AntiOutlier(model, calib_data=calib_data, cfg=anti_config)
        anti_outlier.process()
        print("[INFO] AntiOutlier processing done")

    print(f"[INFO] Configuring W{args.w_bit}A{args.a_bit} quantization...")
    quant_config = QuantConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        disable_names=args.disable_names,
        dev_type=args.device_type,
        dev_id=args.device_id if args.device_type == "npu" else None,
        act_method=args.act_method,
        do_smooth=args.do_smooth,
        open_outlier=True,
        mm_tensor=False,
        w_sym=True,
        disable_last_linear=True,
    )

    print(f"[INFO] Running calibration with {len(calib_data)} samples...")
    calibrator = Calibrator(
        model=model,
        cfg=quant_config,
        calib_data=calib_data,
        disable_level=args.disable_level,
    )
    calibrator.run()

    os.makedirs(args.save_directory, exist_ok=True)
    print(f"[INFO] Saving quantized model to: {args.save_directory}")
    calibrator.save(args.save_directory, save_type=["safe_tensor"])

    tokenizer.save_pretrained(args.save_directory)

    src_config = os.path.join(args.model_path, "config.json")
    dst_config = os.path.join(args.save_directory, "config.json")
    if os.path.exists(src_config) and not os.path.exists(dst_config):
        shutil.copy(src_config, dst_config)

    print("")
    print("[DONE] Quantization complete!")
    print(f"  Model saved to: {args.save_directory}")
    print(f"  Quantization: W{args.w_bit}A{args.a_bit}")
    print(f"  Anti-outlier: {args.anti_method}")
    print(f"  Skipped layers: {args.disable_names}")


if __name__ == "__main__":
    main()
