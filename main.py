"""
Unified entry point for qwen-ascend-llm.

Subcommands:
    serve   - Start the OpenAI-compatible API server
    chat    - Interactive CLI chat
    bench   - Run inference benchmarks

Examples:
    python main.py serve --config configs/deepseek_r1_1.5b_310.json
    python main.py chat --om_model_path ./output/model_310_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_sim.om
    python main.py bench --om_model_path ./output/model_310_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_sim.om
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "serve":
        from server import main as serve_main
        serve_main()
    elif command == "chat":
        import cli_chat
        cli_chat.main_cli()
    elif command == "bench":
        from benchmark import main as bench_main
        bench_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: serve, chat, bench")
        sys.exit(1)


if __name__ == "__main__":
    main()
