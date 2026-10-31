import csv
import ast
import os
import sys
import numpy as np
from collections import defaultdict, Counter

FULL_CSV_PATH = "d:/PBL/borg_traces_data.csv"

def parse_dict(val):
    if not val or val == 'None' or val == '' or val == '[]':
        return None
    try:
        return ast.literal_eval(val)
    except Exception:
        return None

if not os.path.exists(FULL_CSV_PATH):
    print(f"Error: {FULL_CSV_PATH} not found.")
    sys.exit(1)

file_size_mb = os.path.getsize(FULL_CSV_PATH) / (1024 * 1024)
print(f"Analyzing full Borg dataset: {FULL_CSV_PATH} ({file_size_mb:.2f} MB)...")

total_rows = 0
cols = []
seen_row_hashes = set()
duplicate_rows = 0

min_time = float('inf')
max_time = float('-inf')
unique_times = set()

# Grouping dictionaries: store list of (time_us, idx, cpu_avg, cpu_req)
instance_groups = defaultdict(list)
collection_groups = defaultdict(list)
machine_groups = defaultdict(list)

with open(FULL_CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    for idx, r in enumerate(reader):
        total_rows += 1
        
        # Timestamp
        try:
            t = float(r['time'])
            min_time = min(min_time, t)
            max_time = max(max_time, t)
            unique_times.add(t)
        except ValueError:
            t = 0.0

        cid = r['collection_id']
        i_idx = r['instance_index']
        mid = r['machine_id']

        avg = parse_dict(r['average_usage']) or {}
        req = parse_dict(r['resource_request']) or {}

        cpu_avg = float(avg.get('cpus')) if avg.get('cpus') is not None else np.nan
        cpu_req = float(req.get('cpus')) if req.get('cpus') is not None else np.nan

        # Collect for grouping analysis
        if cid and i_idx:
            instance_groups[(cid, i_idx)].append((t, idx, cpu_avg, cpu_req))
        if cid:
            collection_groups[cid].append((t, idx, cpu_avg, cpu_req))
        if mid:
            machine_groups[mid].append((t, idx, cpu_avg, cpu_req))

print(f"\n--- Full Dataset Basic Stats ---")
print(f"Total Rows: {total_rows}")
print(f"Total Columns: {len(cols)} ({cols})")
print(f"Timestamp Min: {min_time:.0f} us, Max: {max_time:.0f} us")
total_span_days = (max_time - min_time) / 1e6 / 86400.0
print(f"Total Temporal Span: {total_span_days:.2f} days")
print(f"Unique Timestamps: {len(unique_times)}")

def group_stats(group_dict, name):
    num_groups = len(group_dict)
    lengths = [len(v) for v in group_dict.values()]
    if not lengths:
        return f"{name}: Empty"
    
    gte_2 = sum(1 for l in lengths if l >= 2)
    gte_5 = sum(1 for l in lengths if l >= 5)
    gte_10 = sum(1 for l in lengths if l >= 10)
    gte_20 = sum(1 for l in lengths if l >= 20)
    gte_50 = sum(1 for l in lengths if l >= 50)
    gte_100 = sum(1 for l in lengths if l >= 100)

    print(f"\n--- Grouping Analysis: {name} ---")
    print(f"Unique Groups: {num_groups}")
    print(f"Obs per group — Min: {min(lengths)}, Median: {np.median(lengths):.1f}, Mean: {np.mean(lengths):.2f}, Max: {max(lengths)}")
    print(f"Groups >= 2: {gte_2} ({gte_2/num_groups*100:.2f}%)")
    print(f"Groups >= 5: {gte_5} ({gte_5/num_groups*100:.2f}%)")
    print(f"Groups >= 10: {gte_10} ({gte_10/num_groups*100:.2f}%)")
    print(f"Groups >= 20: {gte_20} ({gte_20/num_groups*100:.2f}%)")
    print(f"Groups >= 50: {gte_50} ({gte_50/num_groups*100:.2f}%)")
    print(f"Groups >= 100: {gte_100} ({gte_100/num_groups*100:.2f}%)")

group_stats(instance_groups, "(collection_id, instance_index)")
group_stats(collection_groups, "collection_id")
group_stats(machine_groups, "machine_id")

# Contiguous Pair Analysis for Instance Grouping
deltas_inst = []
pairs_le_5m = 0
pairs_le_10m = 0
pairs_le_15m = 0
pairs_le_30m = 0
pairs_le_1h = 0
pairs_le_6h = 0
pairs_le_24h = 0

for key, trace in instance_groups.items():
    if len(trace) >= 2:
        trace.sort(key=lambda x: x[0])
        for i in range(len(trace) - 1):
            dt_sec = (trace[i+1][0] - trace[i][0]) / 1e6
            deltas_inst.append(dt_sec)
            if dt_sec <= 300: pairs_le_5m += 1
            if dt_sec <= 600: pairs_le_10m += 1
            if dt_sec <= 900: pairs_le_15m += 1
            if dt_sec <= 1800: pairs_le_30m += 1
            if dt_sec <= 3600: pairs_le_1h += 1
            if dt_sec <= 21600: pairs_le_6h += 1
            if dt_sec <= 86400: pairs_le_24h += 1

print(f"\n--- Contiguous Time Gap Analysis: (collection_id, instance_index) ---")
print(f"Total Consecutive Pairs: {len(deltas_inst)}")
if deltas_inst:
    print(f"Delta_t stats (sec) — Min: {min(deltas_inst):.1f}, p25: {np.percentile(deltas_inst, 25):.1f}, Median: {np.median(deltas_inst):.1f}, Mean: {np.mean(deltas_inst):.1f}, p75: {np.percentile(deltas_inst, 75):.1f}, p90: {np.percentile(deltas_inst, 90):.1f}, p95: {np.percentile(deltas_inst, 95):.1f}, Max: {max(deltas_inst):.1f}")
    print(f"Pairs <= 5 min: {pairs_le_5m} ({pairs_le_5m/len(deltas_inst)*100:.2f}%)")
    print(f"Pairs <= 10 min: {pairs_le_10m} ({pairs_le_10m/len(deltas_inst)*100:.2f}%)")
    print(f"Pairs <= 15 min: {pairs_le_15m} ({pairs_le_15m/len(deltas_inst)*100:.2f}%)")
    print(f"Pairs <= 30 min: {pairs_le_30m} ({pairs_le_30m/len(deltas_inst)*100:.2f}%)")
    print(f"Pairs <= 1 hour: {pairs_le_1h} ({pairs_le_1h/len(deltas_inst)*100:.2f}%)")
    print(f"Pairs <= 6 hours: {pairs_le_6h} ({pairs_le_6h/len(deltas_inst)*100:.2f}%)")
    print(f"Pairs <= 24 hours: {pairs_le_24h} ({pairs_le_24h/len(deltas_inst)*100:.2f}%)")
