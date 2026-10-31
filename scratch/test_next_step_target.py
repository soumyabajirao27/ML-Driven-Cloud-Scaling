import csv
import ast
from collections import defaultdict
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

# Group by instance
instance_groups = defaultdict(list)
for r in rows:
    cid = r['collection_id']
    idx = r['instance_index']
    t = float(r['start_time']) if r['start_time'] != '' else float(r['time'])
    avg = parse_dict(r['average_usage']) or {}
    req = parse_dict(r['resource_request']) or {}
    cpu_avg = float(avg.get('cpus')) if avg.get('cpus') is not None else np.nan
    cpu_req = float(req.get('cpus')) if req.get('cpus') is not None else np.nan
    
    cpu_util = (cpu_avg / cpu_req * 100.0) if (not np.isnan(cpu_avg) and not np.isnan(cpu_req) and cpu_req > 0) else np.nan
    
    instance_groups[(cid, idx)].append({
        'time_us': t,
        'cpu_avg': cpu_avg,
        'cpu_util_pct': cpu_util,
        'row': r
    })

valid_future_pairs = 0
tight_future_pairs = 0  # delta_t <= 3600s (1 hour)

for key, group in instance_groups.items():
    if len(group) >= 2:
        # Sort chronologically by time_us
        group.sort(key=lambda x: x['time_us'])
        for i in range(len(group) - 1):
            dt_sec = (group[i+1]['time_us'] - group[i]['time_us']) / 1e6
            valid_future_pairs += 1
            if 0 < dt_sec <= 3600:
                tight_future_pairs += 1

print(f"Total Instance Groups: {len(instance_groups)}")
print(f"Total (t -> t+1) pairs in dataset: {valid_future_pairs}")
print(f"Total tight (t -> t+1) pairs with 0 < delta_t <= 1 hour: {tight_future_pairs}")
