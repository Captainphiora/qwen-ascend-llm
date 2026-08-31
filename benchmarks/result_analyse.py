import json, os, glob
from collections import defaultdict

files = glob.glob("results/_tmp_device_*.jsonl") + ["results/math500_results.jsonl"]

records = {}
for path in files:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                records[r["unique_id"]] = r
            except Exception:
                pass

if not records:
    print("No results found.")
else:
    correct_total = 0
    subject_stats = defaultdict(lambda: [0.0, 0])
    level_stats = defaultdict(lambda: [0.0, 0])
    for r in records.values():
        correct_total += r["pass1"]
        subject_stats[r["subject"]][0] += r["pass1"]
        subject_stats[r["subject"]][1] += 1
        level_stats[r["level"]][0] += r["pass1"]
        level_stats[r["level"]][1] += 1

    total = len(records)
    print(f"Finished: {total}/500")
    print(f"Finished: {total}/500")
    print(f"pass@1 so far: {correct_total/total:.4f}  ({correct_total:.1f}/{total})")
    print("\n--- By Subject ---")
    for subj, (c, t) in sorted(subject_stats.items()):
        print(f"  {subj}: {c/t:.4f}  ({c:.1f}/{t})")
    print("\n--- By Level ---")
    for lvl, (c, t) in sorted(level_stats.items()):
        print(f"  Level {lvl}: {c/t:.4f}  ({c:.1f}/{t})")