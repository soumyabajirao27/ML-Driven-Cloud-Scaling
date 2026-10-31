import csv
import ast
import os
import numpy as np
from collections import defaultdict

FULL_CSV_PATH = "d:/PBL/borg_traces_data.csv"

def parse_dict(val):
    if not val or val == 'None' or val == '' or val == '[]':
        return None
    try:
        return ast.literal_eval(val)
    except Exception:
        return None

rows = []
with open(FULL_CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, r in enumerate(reader):
        try:
            t = float(r['time'])
            # Filter out sentinel max int outliers
            if 0 <= t <= 3.0e12:
                rows.append((t, idx, r))
        except ValueError:
            pass

print(f"Loaded {len(rows)} valid timestamp rows.")

# Sort chronologically by time
rows.sort(key=lambda x: x[0])

# Group by (collection_id, instance_index)
instance_traces = defaultdict(list)
for t, idx, r in rows:
    cid = r['collection_id']
    i_idx = r['instance_index']
    if cid and i_idx:
        avg = parse_dict(r['average_usage']) or {}
        req = parse_dict(r['resource_request']) or {}
        cpu_avg = float(avg.get('cpus')) if avg.get('cpus') is not None else np.nan
        cpu_req = float(req.get('cpus')) if req.get('cpus') is not None else np.nan
        cpu_util = (cpu_avg / cpu_req * 100.0) if (not np.isnan(cpu_avg) and not np.isnan(cpu_req) and cpu_req > 0) else np.nan
        
        instance_traces[(cid, i_idx)].append({
            'time_us': t,
            'cpu_avg': cpu_avg,
            'cpu_req': cpu_req,
            'cpu_util': cpu_util,
            'row': r
        })

print(f"Total Unique Instances: {len(instance_traces)}")

# Construct valid 1-step temporal pairs (t -> t+1) with 0 < delta_t <= 3600 sec
pairs_1h = []
pairs_15m = []

for key, trace in instance_traces.items():
    if len(trace) >= 2:
        trace.sort(key=lambda x: x['time_us'])
        for i in range(len(trace) - 1):
            dt_sec = (trace[i+1]['time_us'] - trace[i]['time_us']) / 1e6
            if 0 < dt_sec <= 3600:
                pairs_1h.append((trace[i], trace[i+1], dt_sec))
            if 0 < dt_sec <= 900:  # <= 15 min
                pairs_15m.append((trace[i], trace[i+1], dt_sec))

print(f"\n--- 1-Step Future Target Pairs (t -> t+1) ---")
print(f"Pairs with delta_t <= 1 hour (3600s): {len(pairs_1h)}")
print(f"Pairs with delta_t <= 15 min (900s): {len(pairs_15m)}")

# Construct valid LSTM sliding sequences of length L=5 (dt <= 1 hr between adjacent steps)
seq_l5 = []
for key, trace in instance_traces.items():
    if len(trace) >= 6:  # 5 inputs + 1 target
        trace.sort(key=lambda x: x['time_us'])
        for i in range(len(trace) - 5):
            window = trace[i:i+6]
            # Check if all adjacent time gaps are <= 3600s
            is_valid = True
            for j in range(5):
                dt = (window[j+1]['time_us'] - window[j]['time_us']) / 1e6
                if not (0 < dt <= 3600):
                    is_valid = False
                    break
            if is_valid:
                seq_l5.append(window)

print(f"\n--- LSTM Sliding Sequences (Length L=5 inputs -> 1 target) ---")
print(f"Valid L=5 contiguous sequences (delta_t <= 1 hr): {len(seq_l5)}")

# Test sequence length L=10
seq_l10 = []
for key, trace in instance_traces.items():
    if len(trace) >= 11:
        trace.sort(key=lambda x: x['time_us'])
        for i in range(len(trace) - 10):
            window = trace[i:i+11]
            is_valid = True
            for j in range(10):
                dt = (window[j+1]['time_us'] - window[j]['time_us']) / 1e6
                if not (0 < dt <= 3600):
                    is_valid = False
                    break
            if is_valid:
                seq_l10.append(window)

print(f"Valid L=10 contiguous sequences (delta_t <= 1 hr): {len(seq_l10)}")
