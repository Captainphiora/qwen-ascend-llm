"""
Server entry point: launches the OpenAI-compatible API service.

Usage:
    python server.py --config configs/deepseek_r1_1.5b_310.json
    python server.py --config configs/deepseek_r1_1.5b_310.json --port 8080
"""

import argparse
import json
import math
import os
import sys

import uvicorn


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_server_args():
    parser = argparse.ArgumentParser(description="DeepSeek/Qwen LLM API Server on Ascend NPU")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to model serving config JSON (e.g. configs/deepseek_r1_1.5b_310.json)"
    )
    parser.add_argument("--host", type=str, default=None, help="Override server host")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    return parser.parse_args()


def main():
    args = parse_server_args()
    cfg = load_config(args.config)

    host = args.host or cfg.get("server", {}).get("host", "0.0.0.0")
    port = args.port or cfg.get("server", {}).get("port", 8000)
    workers = cfg.get("server", {}).get("workers", 1)

    os.environ["_SERVING_CONFIG_PATH"] = os.path.abspath(args.config)

    print("=" * 60)
    print(f"  Model : {cfg.get('model_name', 'unknown')}")
    print(f"  Config: {args.config}")
    print(f"  Listen: http://{host}:{port}")
    print("=" * 60)

    uvicorn.run("api:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
