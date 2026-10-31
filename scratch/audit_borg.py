import csv
import ast
from collections import defaultdict, Counter
import numpy as np

CSV_PATH = "d:/PBL/borg_traces_30000.csv"

def parse_dict(val):
    if not val or val == 'None' or val == '' or val == '[]':
        return None
    try:
        return ast.literal_eval(val)
    except Exception:
        return None

rows = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Sort chronologically by time
rows.sort(key=lambda r: float(r['time']) if r['time'] != '' else 0.0)

print(f"Total sorted rows: {len(rows)}")

# 1. Split indices (80% Train, 10% Val, 10% Test)
train_rows = rows[:24000]
val_rows = rows[24000:27000]
test_rows = rows[27000:]

# Extract ID sets
train_colls = set(r['collection_id'] for r in train_rows if r['collection_id'] != '')
val_colls = set(r['collection_id'] for r in val_rows if r['collection_id'] != '')
test_colls = set(r['collection_id'] for r in test_rows if r['collection_id'] != '')

train_insts = set((r['collection_id'], r['instance_index']) for r in train_rows if r['collection_id'] != '' and r['instance_index'] != '')
val_insts = set((r['collection_id'], r['instance_index']) for r in val_rows if r['collection_id'] != '' and r['instance_index'] != '')
test_insts = set((r['collection_id'], r['instance_index']) for r in test_rows if r['collection_id'] != '' and r['instance_index'] != '')

train_machs = set(r['machine_id'] for r in train_rows if r['machine_id'] != '')
val_machs = set(r['machine_id'] for r in val_rows if r['machine_id'] != '')
test_machs = set(r['machine_id'] for r in test_rows if r['machine_id'] != '')

print("\n--- Split Overlap Audit ---")
print(f"Unique Collection IDs — Train: {len(train_colls)}, Val: {len(val_colls)}, Test: {len(test_colls)}")
print(f"Collection ID Overlap (Train & Val): {len(train_colls & val_colls)} ({len(train_colls & val_colls)/len(val_colls)*100:.1f}% of Val)")
print(f"Collection ID Overlap (Train & Test): {len(train_colls & test_colls)} ({len(train_colls & test_colls)/len(test_colls)*100:.1f}% of Test)")

print(f"\nUnique Instance Keys — Train: {len(train_insts)}, Val: {len(val_insts)}, Test: {len(test_insts)}")
print(f"Instance Key Overlap (Train & Val): {len(train_insts & val_insts)} ({len(train_insts & val_insts)/len(val_insts)*100:.1f}% of Val)")
print(f"Instance Key Overlap (Train & Test): {len(train_insts & test_insts)} ({len(train_insts & test_insts)/len(test_insts)*100:.1f}% of Test)")

print(f"\nUnique Machine IDs — Train: {len(train_machs)}, Val: {len(val_machs)}, Test: {len(test_machs)}")
print(f"Machine ID Overlap (Train & Val): {len(train_machs & val_machs)} ({len(train_machs & val_machs)/len(val_machs)*100:.1f}% of Val)")
print(f"Machine ID Overlap (Train & Test): {len(train_machs & test_machs)} ({len(train_machs & test_machs)/len(test_machs)*100:.1f}% of Test)")

# 2. Instance Sequence & Delta Audit
instance_traces = defaultdict(list)
for idx, r in enumerate(rows):
    key = (r['collection_id'], r['instance_index'])
    t = float(r['start_time']) if r['start_time'] != '' else float(r['time'])
    avg = parse_dict(r['average_usage']) or {}
    req = parse_dict(r['resource_request']) or {}
    cpu_avg = float(avg.get('cpus')) if avg.get('cpus') is not None else np.nan
    cpu_req = float(req.get('cpus')) if req.get('cpus') is not None else np.nan
    instance_traces[key].append((t, idx, cpu_avg, cpu_req))

seq_lengths = Counter(len(v) for v in instance_traces.values())

print("\n--- Sequence Length Audit per Instance (collection_id, instance_index) ---")
print(f"Total Unique Instances: {len(instance_traces)}")
print(f"Instances with length == 1: {seq_lengths[1]} ({seq_lengths[1]/len(instance_traces)*100:.2f}%)")
print(f"Instances with length >= 2: {sum(v for k,v in seq_lengths.items() if k >= 2)} ({sum(v for k,v in seq_lengths.items() if k >= 2)/len(instance_traces)*100:.2f}%)")
print(f"Instances with length >= 5: {sum(v for k,v in seq_lengths.items() if k >= 5)} ({sum(v for k,v in seq_lengths.items() if k >= 5)/len(instance_traces)*100:.2f}%)")
print(f"Instances with length >= 10: {sum(v for k,v in seq_lengths.items() if k >= 10)} ({sum(v for k,v in seq_lengths.items() if k >= 10)/len(instance_traces)*100:.2f}%)")

# Calculate time deltas delta_t for instances with >= 2 observations
deltas = []
future_pairs = 0
for key, trace in instance_traces.items():
    if len(trace) >= 2:
        # Sort trace by time
        trace.sort(key=lambda x: x[0])
        for i in range(len(trace) - 1):
            dt_sec = (trace[i+1][0] - trace[i][0]) / 1e6
            deltas.append(dt_sec)
            future_pairs += 1

print(f"\nFuture Target Pairs Construction (t -> t+1):")
print(f"Total valid (t -> t+1) future prediction pairs across 30k dataset: {future_pairs}")
if deltas:
    print(f"Time delta (delta_t) stats: min={min(deltas):.1f}s, median={np.median(deltas):.1f}s ({(np.median(deltas)/60):.1f} min), max={max(deltas):.1f}s ({(max(deltas)/3600):.1f} hrs), avg={np.mean(deltas):.1f}s")
