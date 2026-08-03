"""合并 _tmp_device_*.jsonl 到 math500_results.jsonl（910 结果）"""
import json
import glob
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_math500_k1")
MAIN_FILE = os.path.join(RESULTS_DIR, "math500_results.jsonl")
TMP_PATTERN = os.path.join(RESULTS_DIR, "_tmp_device_*.jsonl")


def load_jsonl(path):
    records = {}
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                records[obj["unique_id"]] = obj
    return records


def main():
    merged = load_jsonl(MAIN_FILE)
    before = len(merged)

    for tmp_file in sorted(glob.glob(TMP_PATTERN)):
        chunk = load_jsonl(tmp_file)
        for uid, obj in chunk.items():
            if uid not in merged:
                merged[uid] = obj

    after = len(merged)
    print(f"合并前: {before} 条，合并后: {after} 条，新增: {after - before} 条")

    with open(MAIN_FILE, "w") as f:
        for obj in merged.values():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"已写入 {MAIN_FILE}")


if __name__ == "__main__":
    main()
