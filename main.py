"""
Unified entry point for qwen-ascend-llm.

Subcommands:
    serve   - Start the OpenAI-compatible API server
    chat    - Interactive CLI chat

Examples:
    python main.py serve --config configs/deepseek_r1_1.5b_310.json
    python main.py chat --om_model_path ./output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om
"""

# curl -X POST http://127.0.0.1:8000/v1/chat/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "DeepSeek-R1-Distill-Qwen-1.5B",
# "messages": [
# {"role": "user", "content": "hi"}
# ],
# "max_tokens": 256,
# "stream": false,
# "do_sample": true,
# "temperature": 0.6,
# "top_p": 0.95
# }'



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
    else:
        print(f"Unknown command: {command}")
        print("Available commands: serve, chat, bench")
        sys.exit(1)


if __name__ == "__main__":
    main()
