import csv
from collections import defaultdict

OUTDIR = "profiling/decode_profile/PROF_000001_20260831191419866_04143416HCCKEABK/mindstudio_profiler_output"
with open(f"{OUTDIR}/op_summary_20260831191821.csv") as f:
    rows = list(csv.DictReader(f))

total_time = sum(float(r.get('Task Duration(us)', 0)) for r in rows)
N = 35

print(f"总 kernel 时间: {total_time/1e6:.3f}s ({len(rows)} kernels, {N} iterations)")
print(f"每 iteration: {total_time/N/1000:.2f}ms")
print()

by_type = defaultdict(lambda: {'count': 0, 'total_us': 0, 'max_us': 0})
for r in rows:
    op = r['OP Type']
    dur = float(r.get('Task Duration(us)', 0))
    by_type[op]['count'] += 1
    by_type[op]['total_us'] += dur
    by_type[op]['max_us'] = max(by_type[op]['max_us'], dur)

header = f"{'OP Type':<30} {'Count':>6} {'Total(ms)':>10} {'Ratio':>7} {'Avg(ms)':>8} {'Max(ms)':>8} {'Per-Iter':>10}"
print(header)
print("-" * 90)
for op, d in sorted(by_type.items(), key=lambda x: -x[1]['total_us']):
    ratio = d['total_us'] / total_time * 100
    avg = d['total_us'] / d['count'] / 1000
    per_iter = d['total_us'] / N / 1000
    print(f"{op:<30} {d['count']:>6} {d['total_us']/1000:>10.2f} {ratio:>6.1f}% {avg:>8.3f} {d['max_us']/1000:>8.3f} {per_iter:>10.3f}")

print()

# BatchMatMulV2 breakdown
bmm_rows = [r for r in rows if r['OP Type'] == 'BatchMatMulV2']
print(f"=== BatchMatMulV2 ({len(bmm_rows)} kernels, {sum(float(r['Task Duration(us)']) for r in bmm_rows)/1000:.2f}ms total) ===")

categories = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
cat_stats = defaultdict(lambda: {'count': 0, 'total_us': 0})

for r in bmm_rows:
    name = r.get('Op Name', '')
    dur = float(r.get('Task Duration(us)', 0))
    found = False
    for cat in categories:
        if cat in name:
            cat_stats[cat]['count'] += 1
            cat_stats[cat]['total_us'] += dur
            found = True
            break
    if not found:
        cat_stats['attn_score']['count'] += 1
        cat_stats['attn_score']['total_us'] += dur

print(f"  {'Category':<15} {'Count':>6} {'Total(ms)':>10} {'Per-Iter':>10} {'Ratio':>7}")
print("  " + "-" * 55)
bmm_total = sum(float(r['Task Duration(us)']) for r in bmm_rows)
for cat in categories + ['attn_score']:
    d = cat_stats[cat]
    if d['count'] > 0:
        per_iter = d['total_us'] / N / 1000
        ratio = d['total_us'] / bmm_total * 100
        print(f"  {cat:<15} {d['count']:>6} {d['total_us']/1000:>10.2f} {per_iter:>10.3f} {ratio:>6.1f}%")

print()

# Transpose
transpose_rows = [r for r in rows if r['OP Type'] == 'Transpose']
print(f"=== Transpose ({len(transpose_rows)} kernels) ===")
for r in transpose_rows[:3]:
    dur = float(r.get('Task Duration(us)', 0))
    name = r.get('Op Name', '')[:80]
    shapes = r.get('Input Shapes', '')[:60]
    print(f"  {dur/1000:.3f}ms | {shapes} | {name[:60]}")
total_tr = sum(float(r['Task Duration(us)']) for r in transpose_rows)
print(f"  Total: {total_tr/1000:.2f}ms, Per-Iter: {total_tr/N/1000:.2f}ms")
print()

# GatherV2 analysis
gather_rows = [r for r in rows if r['OP Type'] == 'GatherV2']
print(f"=== GatherV2 ({len(gather_rows)} kernels) ===")
gather_by_name = defaultdict(lambda: {'count': 0, 'total_us': 0})
for r in gather_rows:
    name = r.get('Op Name', '')
    dur = float(r.get('Task Duration(us)', 0))
    if 'embed_tokens' in name:
        gather_by_name['embed_tokens']['count'] += 1
        gather_by_name['embed_tokens']['total_us'] += dur
    elif 'kv' in name.lower() or 'cache' in name.lower() or 'Concat' in name:
        gather_by_name['kv_related']['count'] += 1
        gather_by_name['kv_related']['total_us'] += dur
    else:
        gather_by_name['other']['count'] += 1
        gather_by_name['other']['total_us'] += dur

for cat, d in gather_by_name.items():
    print(f"  {cat}: count={d['count']}, total={d['total_us']/1000:.2f}ms, per-iter={d['total_us']/N/1000:.2f}ms")

# Show a few GatherV2 op names for context
print("  Sample names:")
seen = set()
for r in gather_rows:
    name = r.get('Op Name', '')[:100]
    short = name.split('/')[-1] if '/' in name else name
    if short not in seen:
        seen.add(short)
        shapes = r.get('Input Shapes', '')[:60]
        print(f"    {short[:40]:<40} {shapes}")
    if len(seen) >= 8:
        break

print()

# Per-iteration time breakdown summary
print("=== Per-Iteration (每 token) 耗时分解 ===")
print(f"  Total per token:   {total_time/N/1000:.2f}ms")
for op, d in sorted(by_type.items(), key=lambda x: -x[1]['total_us'])[:8]:
    per_iter = d['total_us'] / N / 1000
    ratio = d['total_us'] / total_time * 100
    print(f"  {op:<25} {per_iter:>8.2f}ms ({ratio:>5.1f}%)")
