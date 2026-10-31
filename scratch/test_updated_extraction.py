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
            if 0 <= t <= 3.0e12:
                rows.append((t, idx, r))
        except ValueError:
            pass

rows.sort(key=lambda x: x[0])

instance_groups = defaultdict(list)
for t, idx, r in rows:
    cid = r['collection_id']
    i_idx = r['instance_index']
    if cid and i_idx:
        avg = parse_dict(r['average_usage']) or {}
        req = parse_dict(r['resource_request']) or {}
        cpu_avg = float(avg.get('cpus')) if avg.get('cpus') is not None else np.nan
        cpu_req = float(req.get('cpus')) if req.get('cpus') is not None else np.nan
        cpu_util = (cpu_avg / cpu_req * 100.0) if (not np.isnan(cpu_avg) and not np.isnan(cpu_req) and cpu_req > 0) else np.nan
        
        instance_groups[(cid, i_idx)].append({
            'time_us': t,
            'priority': float(r['priority']) if r['priority'] != '' else 0.0,
            'scheduling_class': float(r['scheduling_class']) if r['scheduling_class'] != '' else 0.0,
            'cpu_request': cpu_req,
            'memory_request': float(req.get('memory')) if req.get('memory') is not None else np.nan,
            'assigned_memory': float(r['assigned_memory']) if r['assigned_memory'] != '' else np.nan,
            'cpu_avg': cpu_avg,
            'cpu_util': cpu_util
        })

# Extract 3D sequences with dt <= 1800s (30 min) and time-gap feature
seq_30m = []
for key, trace in instance_groups.items():
    if len(trace) >= 6:
        trace.sort(key=lambda x: x['time_us'])
        for i in range(len(trace) - 5):
            window = trace[i:i+6]
            is_valid = True
            step_gaps = []
            for j in range(5):
                dt = (window[j+1]['time_us'] - window[j]['time_us']) / 1e6
                if not (0 < dt <= 1800):  # Strict 30-min window
                    is_valid = False
                    break
                step_gaps.append(dt)
            
            if is_valid:
                seq_30m.append((window, step_gaps))

print(f"Total Strict (<=30 min) LSTM Sequences Extracted: {len(seq_30m):,}")
