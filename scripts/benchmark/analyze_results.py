"""分析 math500_results.jsonl 的总体结果，输出 by subject 和 by level 准确率"""
import json
import os
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_math500_k1")
MAIN_FILE = os.path.join(RESULTS_DIR, "math500_results.jsonl")


def accuracy(correct, total):
    return correct / total if total else 0.0


def main():
    by_subject = defaultdict(lambda: [0, 0])
    by_level = defaultdict(lambda: [0, 0])
    total_correct = 0
    total = 0

    with open(MAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            correct = obj["pass1"] >= 0.5
            subj = obj["subject"]
            lvl = f"Level {obj['level']}"
            by_subject[subj][0] += int(correct)
            by_subject[subj][1] += 1
            by_level[lvl][0] += int(correct)
            by_level[lvl][1] += 1
            total_correct += int(correct)
            total += 1

    print(f"=== pass@1 (k=1): {accuracy(total_correct, total):.4f}  ({total_correct}/{total}) ===")

    print("\n--- By Subject ---")
    for subj in sorted(by_subject):
        c, n = by_subject[subj]
        print(f"  {subj}: {accuracy(c, n):.4f}  ({c}/{n})")

    print("\n--- By Level ---")
    for lvl in sorted(by_level):
        c, n = by_level[lvl]
        print(f"  {lvl}: {accuracy(c, n):.4f}  ({c}/{n})")

    # 写入 metrics 文件
    out_path = os.path.join(RESULTS_DIR, "math500_metrics.txt")
    with open(out_path, "w") as f:
        f.write(f"=== pass@1 (k=1): {accuracy(total_correct, total):.4f}  ({total_correct}/{total}) ===\n\n")
        f.write("--- By Subject ---\n")
        for subj in sorted(by_subject):
            c, n = by_subject[subj]
            f.write(f"  {subj}: {accuracy(c, n):.4f}  ({c}/{n})\n")
        f.write("\n--- By Level ---\n")
        for lvl in sorted(by_level):
            c, n = by_level[lvl]
            f.write(f"  {lvl}: {accuracy(c, n):.4f}  ({c}/{n})\n")
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
