import csv
import ast
import math
import json
from collections import Counter

CSV_PATH = "d:/PBL/borg_traces_30000.csv"

def parse_dict(val):
    if not val or val == 'None' or val == '':
        return None
    try:
        return ast.literal_eval(val)
    except Exception:
        return None

def parse_list(val):
    if not val or val == 'None' or val == '':
        return []
    try:
        # Some arrays are numpy string repr like "[0.00314 0.00381 ...]"
        val_clean = val.strip()
        if val_clean.startswith('[') and val_clean.endswith(']'):
            inner = val_clean[1:-1].strip()
            if not inner:
                return []
            # Split by whitespace or commas
            tokens = inner.replace(',', ' ').split()
            return [float(t) for t in tokens if t != '']
    except Exception:
        pass
    return []

print("Starting deep inspection of borg_traces_30000.csv...")

total_rows = 0
cols = []
missing_counts = {}
data_types = {}
value_samples = {}
duplicate_rows = 0
seen_rows = set()

# Metrics collection
times = []
start_times = []
end_times = []
durations = []

collections = Counter()
machines = Counter()
instance_indices = Counter()
events = Counter()
failures = Counter()

cpus_req = []
mem_req = []
cpus_avg = []
mem_avg = []
cpus_max = []
mem_max = []
cpus_rand = []
mem_rand = []
assigned_mems = []
page_cache_mems = []

malformed_dicts = 0
malformed_arrays = 0

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    for col in cols:
        missing_counts[col] = 0
        value_samples[col] = []
    
    for row in reader:
        total_rows += 1
        
        # Check duplicate row hash
        row_tuple = tuple(row.items())
        if row_tuple in seen_rows:
            duplicate_rows += 1
        else:
            seen_rows.add(row_tuple)

        for col, val in row.items():
            if val is None or val == '' or val == 'None' or val == '[]':
                missing_counts[col] += 1
            elif len(value_samples[col]) < 5:
                value_samples[col].append(val)

        # Parse Timestamps
        try:
            t = float(row['time']) if row['time'] != '' else None
            st = float(row['start_time']) if row['start_time'] != '' else None
            et = float(row['end_time']) if row['end_time'] != '' else None
            if t is not None: times.append(t)
            if st is not None: start_times.append(st)
            if et is not None: end_times.append(et)
            if st is not None and et is not None:
                durations.append(et - st)
        except ValueError:
            pass

        # Parse Identifiers & Categoricals
        if row['collection_id']: collections[row['collection_id']] += 1
        if row['machine_id']: machines[row['machine_id']] += 1
        if row['instance_index']: instance_indices[row['instance_index']] += 1
        if row['event']: events[row['event']] += 1
        if row['failed']: failures[row['failed']] += 1

        # Parse Dicts
        req = parse_dict(row['resource_request'])
        if req and isinstance(req, dict):
            if req.get('cpus') is not None: cpus_req.append(float(req['cpus']))
            if req.get('memory') is not None: mem_req.append(float(req['memory']))
        elif row['resource_request'] and row['resource_request'] != '[]':
            malformed_dicts += 1

        avg = parse_dict(row['average_usage'])
        if avg and isinstance(avg, dict):
            if avg.get('cpus') is not None: cpus_avg.append(float(avg['cpus']))
            if avg.get('memory') is not None: mem_avg.append(float(avg['memory']))
        
        mx = parse_dict(row['maximum_usage'])
        if mx and isinstance(mx, dict):
            if mx.get('cpus') is not None: cpus_max.append(float(mx['cpus']))
            if mx.get('memory') is not None: mem_max.append(float(mx['memory']))

        rnd = parse_dict(row['random_sample_usage'])
        if rnd and isinstance(rnd, dict):
            if rnd.get('cpus') is not None: cpus_rand.append(float(rnd['cpus']))
            if rnd.get('memory') is not None: mem_rand.append(float(rnd['memory']))

        # Parse scalar numbers
        try:
            if row['assigned_memory'] != '': assigned_mems.append(float(row['assigned_memory']))
            if row['page_cache_memory'] != '': page_cache_mems.append(float(row['page_cache_memory']))
        except ValueError:
            pass

print(f"Total Rows: {total_rows}")
print(f"Total Columns: {len(cols)}")
print(f"Duplicate Exact Rows: {duplicate_rows}")

def stats(arr, name):
    if not arr:
        return f"{name}: empty"
    return f"{name}: count={len(arr)}, min={min(arr):.6f}, max={max(arr):.6f}, avg={sum(arr)/len(arr):.6f}"

print("\n--- Numerical Ranges ---")
print(stats(times, "time"))
print(stats(start_times, "start_time"))
print(stats(end_times, "end_time"))
print(stats(durations, "duration (end-start)"))

print(stats(cpus_req, "cpus_req"))
print(stats(mem_req, "mem_req"))
print(stats(cpus_avg, "cpus_avg"))
print(stats(mem_avg, "mem_avg"))
print(stats(cpus_max, "cpus_max"))
print(stats(mem_max, "mem_max"))
print(stats(cpus_rand, "cpus_rand"))
print(stats(mem_rand, "mem_rand"))
print(stats(assigned_mems, "assigned_mems"))
print(stats(page_cache_mems, "page_cache_mems"))

print("\n--- Categorical Cardinalities ---")
print(f"Unique collection_id: {len(collections)}")
print(f"Unique machine_id: {len(machines)}")
print(f"Unique instance_index: {len(instance_indices)}")
print(f"Events: {dict(events)}")
print(f"Failed counts: {dict(failures)}")

print("\n--- Missing Values Summary ---")
for col, cnt in missing_counts.items():
    pct = (cnt / total_rows) * 100
    print(f"{col}: {cnt} missing ({pct:.2f}%)")
